import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:first_try/core/api/api_interceptors.dart';
import 'package:first_try/core/utils/app_url.dart';
import 'package:first_try/features/auth/current_user.dart';
import 'package:flutter/material.dart';
import 'package:open_file/open_file.dart';
import 'package:path_provider/path_provider.dart';

class TeacherQuestionGeneratorScreen extends StatefulWidget {
  const TeacherQuestionGeneratorScreen({super.key});

  @override
  State<TeacherQuestionGeneratorScreen> createState() =>
      _TeacherQuestionGeneratorScreenState();
}

class _TeacherQuestionGeneratorScreenState
    extends State<TeacherQuestionGeneratorScreen> {
  // ── Picked file ──────────────────────────────────────────────────────────────
  PlatformFile? _pickedFile;

  // ── Form values ──────────────────────────────────────────────────────────────
  int _mcqCount          = 3;
  int _trueFalseCount    = 2;
  int _shortAnswerCount  = 1;
  int _fillBlankCount    = 1;
  int _essayCount        = 0;
  String _difficulty     = 'medium';
  String _language       = 'auto';
  String _title          = 'Exam Questions';
  bool _includeAnswers   = true;
  final Set<String> _bloomLevels = {'remember', 'understand', 'apply'};

  // ── State ────────────────────────────────────────────────────────────────────
  bool _loading = false;
  double _progress = 0;
  String _statusText = '';

  int get _totalQuestions =>
      _mcqCount + _trueFalseCount + _shortAnswerCount + _fillBlankCount + _essayCount;

  // ── Actions ──────────────────────────────────────────────────────────────────

  Future<void> _pickPdf() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
      withData: false,
      withReadStream: false,
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() => _pickedFile = result.files.first);
    }
  }

  Future<void> _generate() async {
    if (_pickedFile == null) return;
    if (_totalQuestions == 0) {
      _showSnack('Set at least one question count.');
      return;
    }
    if (_bloomLevels.isEmpty) {
      _showSnack('Select at least one Bloom\'s level.');
      return;
    }

    setState(() {
      _loading    = true;
      _progress   = 0;
      _statusText = 'Uploading PDF…';
    });

    try {
      final teacherId = context.currentRoleId;
      final url       = AppUrl.teacherGenerateExam(teacherId);

      // Build a dedicated Dio instance with a long timeout for AI generation.
      // We cannot reuse DioConsumer because its receiveTimeout is 10 s.
      final dio = Dio(BaseOptions(
        baseUrl: baseUrl,
        connectTimeout: const Duration(seconds: 15),
        sendTimeout:    const Duration(minutes: 30),
        receiveTimeout: const Duration(minutes: 30),
      ));
      dio.interceptors.add(ApiInterceptor()); // attaches Bearer token

      final formData = FormData.fromMap({
        'pdf':                MultipartFile.fromFileSync(
          _pickedFile!.path!,
          filename: _pickedFile!.name,
        ),
        'mcq_count':          _mcqCount.toString(),
        'true_false_count':   _trueFalseCount.toString(),
        'short_answer_count': _shortAnswerCount.toString(),
        'fill_blank_count':   _fillBlankCount.toString(),
        'essay_count':        _essayCount.toString(),
        'difficulty':         _difficulty,
        'language':           _language,
        'title':              _title,
        'bloom_levels':       _bloomLevels.join(','),
        'include_answers':    _includeAnswers ? 'true' : 'false',
        'output':             'docx',
      });

      setState(() => _statusText = 'Generating questions (this may take a few minutes)…');

      final response = await dio.post<List<int>>(
        url,
        data: formData,
        options: Options(responseType: ResponseType.bytes),
        onSendProgress: (sent, total) {
          if (total > 0) {
            setState(() {
              _progress   = sent / total * 0.1; // upload = first 10%
              _statusText = 'Uploading… ${(sent / 1024 / 1024).toStringAsFixed(1)} MB';
            });
          }
        },
        onReceiveProgress: (received, total) {
          setState(() {
            _progress   = 0.1 + (total > 0 ? received / total * 0.9 : 0);
            _statusText = 'Receiving file…';
          });
        },
      );

      if (response.statusCode == 200 && response.data != null) {
        setState(() => _statusText = 'Saving file…');
        await _saveAndOpen(response.data!);
      } else {
        _showSnack('Generation failed (status ${response.statusCode}).');
      }
    } on DioException catch (e) {
      String msg;
      final data = e.response?.data;
      if (data is List<int> && data.isNotEmpty) {
        // Error body comes back as bytes when responseType == bytes.
        try {
          final json = jsonDecode(utf8.decode(data)) as Map<String, dynamic>;
          msg = json['message'] ?? json['detail'] ?? 'Generation failed.';
        } catch (_) {
          msg = 'Generation failed (status ${e.response?.statusCode}).';
        }
      } else if (e.type == DioExceptionType.sendTimeout ||
                 e.type == DioExceptionType.receiveTimeout ||
                 e.type == DioExceptionType.connectionTimeout) {
        msg = 'Request timed out. The model may still be loading — try again in a minute.';
      } else if (e.type == DioExceptionType.connectionError) {
        msg = 'Cannot reach the server. Check that Laravel and the Python service are running.';
      } else {
        msg = 'Request failed (${e.message}).';
      }
      _showSnack(msg);
    } catch (e) {
      _showSnack('Unexpected error: $e');
    } finally {
      setState(() {
        _loading    = false;
        _progress   = 0;
        _statusText = '';
      });
    }
  }

  Future<void> _saveAndOpen(List<int> bytes) async {
    final dir      = Platform.isAndroid
        ? Directory('/storage/emulated/0/Download')
        : await getApplicationDocumentsDirectory();

    final safeTitle = _title.replaceAll(RegExp(r'[^\w\s]'), '').trim();
    final filename  = '${safeTitle.isEmpty ? "exam" : safeTitle}'
        '_${DateTime.now().millisecondsSinceEpoch}.docx';
    final file = File('${dir.path}/$filename');
    await file.writeAsBytes(bytes);

    if (!mounted) return;
    _showSnack('Saved: $filename');

    await OpenFile.open(file.path);
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  // ── Build ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final cs     = Theme.of(context).colorScheme;
    final purple = const Color(0xFF6366F1);

    return Scaffold(
      backgroundColor: cs.surfaceContainerLowest,
      appBar: AppBar(
        title: const Text('Question Generator'),
        backgroundColor: purple,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Notice ───────────────────────────────────────────────────────
            _InfoBanner(
              icon: Icons.info_outline_rounded,
              text: 'Requires Ollama running with the command-r7b-arabic model pulled. '
                  'Generation takes 2–10 min depending on question count.',
              color: purple,
            ),
            const SizedBox(height: 16),

            // ── PDF picker ───────────────────────────────────────────────────
            _SectionCard(
              title: '1. Upload PDF',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  OutlinedButton.icon(
                    onPressed: _loading ? null : _pickPdf,
                    icon: const Icon(Icons.picture_as_pdf_rounded),
                    label: Text(_pickedFile == null
                        ? 'Choose PDF file'
                        : _pickedFile!.name),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                  ),
                  if (_pickedFile != null) ...[
                    const SizedBox(height: 8),
                    Text(
                      '${(_pickedFile!.size / 1024 / 1024).toStringAsFixed(1)} MB',
                      style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 12),

            // ── Question counts ──────────────────────────────────────────────
            _SectionCard(
              title: '2. Question Types  (Total: $_totalQuestions)',
              child: Column(
                children: [
                  _CountRow(label: 'Multiple Choice (MCQ)', value: _mcqCount,
                      onChanged: (v) => setState(() => _mcqCount = v)),
                  _CountRow(label: 'True / False', value: _trueFalseCount,
                      onChanged: (v) => setState(() => _trueFalseCount = v)),
                  _CountRow(label: 'Short Answer', value: _shortAnswerCount,
                      onChanged: (v) => setState(() => _shortAnswerCount = v)),
                  _CountRow(label: 'Fill in the Blank', value: _fillBlankCount,
                      onChanged: (v) => setState(() => _fillBlankCount = v)),
                  _CountRow(label: 'Essay', value: _essayCount,
                      onChanged: (v) => setState(() => _essayCount = v),
                      max: 20),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // ── Settings ─────────────────────────────────────────────────────
            _SectionCard(
              title: '3. Settings',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Difficulty
                  _DropdownRow(
                    label: 'Difficulty',
                    value: _difficulty,
                    items: const [
                      DropdownMenuItem(value: 'easy',   child: Text('Easy')),
                      DropdownMenuItem(value: 'medium', child: Text('Medium')),
                      DropdownMenuItem(value: 'hard',   child: Text('Hard')),
                    ],
                    onChanged: (v) => setState(() => _difficulty = v!),
                  ),
                  const SizedBox(height: 8),
                  // Language
                  _DropdownRow(
                    label: 'Language',
                    value: _language,
                    items: const [
                      DropdownMenuItem(value: 'auto', child: Text('Auto Detect')),
                      DropdownMenuItem(value: 'ar',   child: Text('Arabic')),
                      DropdownMenuItem(value: 'en',   child: Text('English')),
                    ],
                    onChanged: (v) => setState(() => _language = v!),
                  ),
                  const SizedBox(height: 8),
                  // Title
                  TextField(
                    decoration: const InputDecoration(
                      labelText: 'Exam Title',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    controller: TextEditingController(text: _title),
                    onChanged: (v) => _title = v.isEmpty ? 'Exam Questions' : v,
                  ),
                  const SizedBox(height: 12),
                  const Divider(height: 1),
                  const SizedBox(height: 12),

                  // Bloom's levels
                  Text("Bloom's Taxonomy Levels",
                      style: Theme.of(context).textTheme.bodyMedium
                          ?.copyWith(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 4,
                    children: _kBloomLevels.entries.map((e) {
                      final selected = _bloomLevels.contains(e.key);
                      return FilterChip(
                        label: Text(e.value),
                        selected: selected,
                        onSelected: (on) => setState(() {
                          if (on) {
                            _bloomLevels.add(e.key);
                          } else if (_bloomLevels.length > 1) {
                            _bloomLevels.remove(e.key);
                          }
                        }),
                        selectedColor: purple.withValues(alpha: 0.18),
                        checkmarkColor: purple,
                        labelStyle: TextStyle(
                          color: selected ? purple : cs.onSurfaceVariant,
                          fontWeight:
                              selected ? FontWeight.w600 : FontWeight.normal,
                        ),
                      );
                    }).toList(),
                  ),
                  const SizedBox(height: 12),

                  // Include answers
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Include Answer Key'),
                    subtitle: const Text('Appends answers at the end of the Word file'),
                    value: _includeAnswers,
                    onChanged: (v) => setState(() => _includeAnswers = v),
                    activeColor: purple,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── Progress / Generate button ────────────────────────────────────
            if (_loading) ...[
              LinearProgressIndicator(
                value: _progress > 0 ? _progress : null,
                color: purple,
                backgroundColor: purple.withValues(alpha: 0.15),
              ),
              const SizedBox(height: 8),
              Text(
                _statusText,
                style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
            ],

            FilledButton.icon(
              onPressed: (_loading || _pickedFile == null || _totalQuestions == 0)
                  ? null
                  : _generate,
              icon: _loading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.download_rounded),
              label: Text(_loading
                  ? 'Generating… please wait'
                  : 'Generate & Download (.docx)'),
              style: FilledButton.styleFrom(
                backgroundColor: purple,
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}

// ── Constants ─────────────────────────────────────────────────────────────────

const _kBloomLevels = {
  'remember':  'Remember',
  'understand':'Understand',
  'apply':     'Apply',
  'analyze':   'Analyze',
  'evaluate':  'Evaluate',
  'create':    'Create',
};

// ── Small reusable widgets ────────────────────────────────────────────────────

class _InfoBanner extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color color;
  const _InfoBanner({required this.icon, required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text,
                style: TextStyle(color: color, fontSize: 13)),
          ),
        ],
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget child;
  const _SectionCard({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.outlineVariant),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .titleSmall
                  ?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 14),
          child,
        ],
      ),
    );
  }
}

class _CountRow extends StatelessWidget {
  final String label;
  final int value;
  final ValueChanged<int> onChanged;
  final int max;
  const _CountRow({
    required this.label,
    required this.value,
    required this.onChanged,
    this.max = 50,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
              child: Text(label,
                  style: const TextStyle(fontSize: 13))),
          IconButton(
            icon: const Icon(Icons.remove_circle_outline, size: 20),
            onPressed: value > 0 ? () => onChanged(value - 1) : null,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
          SizedBox(
            width: 32,
            child: Text('$value',
                textAlign: TextAlign.center,
                style: TextStyle(
                    fontWeight: FontWeight.w700,
                    color: cs.primary)),
          ),
          IconButton(
            icon: const Icon(Icons.add_circle_outline, size: 20),
            onPressed: value < max ? () => onChanged(value + 1) : null,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
        ],
      ),
    );
  }
}

class _DropdownRow extends StatelessWidget {
  final String label;
  final String value;
  final List<DropdownMenuItem<String>> items;
  final ValueChanged<String?> onChanged;
  const _DropdownRow({
    required this.label,
    required this.value,
    required this.items,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 90,
          child: Text(label,
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
        ),
        Expanded(
          child: DropdownButtonFormField<String>(
            value: value,
            items: items,
            onChanged: onChanged,
            isDense: true,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            ),
          ),
        ),
      ],
    );
  }
}
