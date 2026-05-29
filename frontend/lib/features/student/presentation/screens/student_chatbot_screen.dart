import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/features/student/presentation/cubit/chatbot_cubit.dart';
import 'package:first_try/features/student/presentation/cubit/chatbot_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

const _kGradient = [Color(0xFF3B82F6), Color(0xFF6366F1)];

class StudentChatbotScreen extends StatefulWidget {
  const StudentChatbotScreen({super.key});

  @override
  State<StudentChatbotScreen> createState() => _StudentChatbotScreenState();
}

class _StudentChatbotScreenState extends State<StudentChatbotScreen> {
  final _controller = TextEditingController();
  final _scrollController = ScrollController();
  late final ChatbotCubit _cubit;
  String _selectedGrade = '';
  String _selectedSubject = '';

  static const _grades = ['', '10', '11', '12'];
  static const _subjects = ['', 'فيزياء', 'كيمياء', 'رياضيات'];
  static const _gradeLabels = ['All grades', 'Grade 10', 'Grade 11', 'Grade 12'];
  static const _subjectLabels = ['All subjects', 'Physics', 'Chemistry', 'Math'];

  @override
  void initState() {
    super.initState();
    _cubit = ChatbotCubit();
  }

  @override
  void dispose() {
    _cubit.close();
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _send() {
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    _controller.clear();
    _cubit.send(text, grade: _selectedGrade, subject: _selectedSubject);
    _scrollToBottom(delayed: true);
  }

  void _scrollToBottom({bool delayed = false}) {
    Future.delayed(Duration(milliseconds: delayed ? 300 : 0), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider.value(
      value: _cubit,
      child: Builder(builder: (context) => _buildScaffold(context)),
    );
  }

  Widget _buildScaffold(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: cs.surfaceContainerLowest,
      body: Column(
        children: [
          _Header(
            selectedGrade: _selectedGrade,
            selectedSubject: _selectedSubject,
            grades: _grades,
            subjects: _subjects,
            gradeLabels: _gradeLabels,
            subjectLabels: _subjectLabels,
            onGradeChanged: (v) => setState(() => _selectedGrade = v),
            onSubjectChanged: (v) => setState(() => _selectedSubject = v),
            onClear: () => context.read<ChatbotCubit>().clearChat(),
          ),
          Expanded(
            child: BlocConsumer<ChatbotCubit, ChatbotState>(
              listener: (context, state) {
                if (state is! ChatbotSending) _scrollToBottom(delayed: true);
              },
              builder: (context, state) {
                if (state.messages.isEmpty) {
                  return _EmptyState();
                }
                return ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                  itemCount: state.messages.length +
                      (state is ChatbotSending ? 1 : 0) +
                      (state is ChatbotError ? 1 : 0),
                  itemBuilder: (context, i) {
                    if (i < state.messages.length) {
                      return _MessageBubble(message: state.messages[i]);
                    }
                    if (state is ChatbotSending) return const _TypingIndicator();
                    if (state is ChatbotError) return _ErrorBanner(state.error);
                    return const SizedBox.shrink();
                  },
                );
              },
            ),
          ),
          _InputBar(controller: _controller, onSend: _send),
        ],
      ),
    );
  }
}

// ── Header ────────────────────────────────────────────────────────────────────

class _Header extends StatelessWidget {
  final String selectedGrade;
  final String selectedSubject;
  final List<String> grades;
  final List<String> subjects;
  final List<String> gradeLabels;
  final List<String> subjectLabels;
  final ValueChanged<String> onGradeChanged;
  final ValueChanged<String> onSubjectChanged;
  final VoidCallback onClear;

  const _Header({
    required this.selectedGrade,
    required this.selectedSubject,
    required this.grades,
    required this.subjects,
    required this.gradeLabels,
    required this.subjectLabels,
    required this.onGradeChanged,
    required this.onSubjectChanged,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.of(context).padding.top;
    return Container(
      padding: EdgeInsets.fromLTRB(16, top + 12, 16, 12),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: _kGradient,
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF3B82F6).withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.2),
                borderRadius: Radii.smRadius,
              ),
              child: const Icon(Icons.auto_awesome_rounded,
                  color: Colors.white, size: 20),
            ),
            const SizedBox(width: 10),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'AI Study Assistant',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    'Ask anything from your curriculum',
                    style: TextStyle(
                        color: Colors.white70,
                        fontSize: 12),
                  ),
                ],
              ),
            ),
            IconButton(
              onPressed: onClear,
              icon: const Icon(Icons.refresh_rounded,
                  color: Colors.white70, size: 20),
              tooltip: 'Clear chat',
            ),
          ]),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(
              child: _FilterChip(
                label: gradeLabels[grades.indexOf(selectedGrade)],
                icon: Icons.school_outlined,
                items: List.generate(
                    grades.length, (i) => DropdownMenuItem(value: grades[i], child: Text(gradeLabels[i]))),
                value: selectedGrade,
                onChanged: (v) => onGradeChanged(v ?? ''),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _FilterChip(
                label: subjectLabels[subjects.indexOf(selectedSubject)],
                icon: Icons.menu_book_outlined,
                items: List.generate(
                    subjects.length, (i) => DropdownMenuItem(value: subjects[i], child: Text(subjectLabels[i]))),
                value: selectedSubject,
                onChanged: (v) => onSubjectChanged(v ?? ''),
              ),
            ),
          ]),
        ],
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final List<DropdownMenuItem<String>> items;
  final String value;
  final ValueChanged<String?> onChanged;

  const _FilterChip({
    required this.label,
    required this.icon,
    required this.items,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.3)),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value,
          items: items,
          onChanged: onChanged,
          icon: const Icon(Icons.arrow_drop_down, color: Colors.white70, size: 18),
          style: const TextStyle(color: Colors.white, fontSize: 12),
          dropdownColor: const Color(0xFF4F46E5),
          isDense: true,
          isExpanded: true,
          selectedItemBuilder: (_) => items
              .map((_) => Row(children: [
                    Icon(icon, color: Colors.white70, size: 14),
                    const SizedBox(width: 4),
                    Text(label,
                        style: const TextStyle(
                            color: Colors.white, fontSize: 12)),
                  ]))
              .toList(),
        ),
      ),
    );
  }
}

// ── Empty state ───────────────────────────────────────────────────────────────

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: _kGradient
                      .map((c) => c.withValues(alpha: 0.12))
                      .toList(),
                ),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.auto_awesome_rounded,
                  color: Color(0xFF3B82F6), size: 36),
            ),
            const SizedBox(height: 16),
            Text(
              'Ask me anything!',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: cs.onSurface,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              'Physics, Chemistry and Math\nfrom the Syrian curriculum (grades 10–12).',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: cs.onSurfaceVariant,
                  ),
            ),
            const SizedBox(height: 24),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: const [
                _SuggestionChip('ما هو قانون نيوتن الثاني؟'),
                _SuggestionChip('اشرح لي تفاعلات الأكسدة والاختزال'),
                _SuggestionChip('كيف أحل معادلة تربيعية؟'),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SuggestionChip extends StatelessWidget {
  final String label;
  const _SuggestionChip(this.label);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => context.read<ChatbotCubit>().send(label),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF3B82F6).withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
              color: const Color(0xFF3B82F6).withValues(alpha: 0.3)),
        ),
        child: Text(
          label,
          style: const TextStyle(
              fontSize: 12,
              color: Color(0xFF3B82F6),
              fontWeight: FontWeight.w500),
          textDirection: TextDirection.rtl,
        ),
      ),
    );
  }
}

// ── Message bubble ────────────────────────────────────────────────────────────

class _MessageBubble extends StatelessWidget {
  final ChatMessage message;
  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isUser = message.isUser;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isUser) ...[
            Container(
              width: 28,
              height: 28,
              decoration: const BoxDecoration(
                gradient: LinearGradient(colors: _kGradient),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.auto_awesome_rounded,
                  color: Colors.white, size: 14),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: isUser
                  ? CrossAxisAlignment.end
                  : CrossAxisAlignment.start,
              children: [
                if (message.isOffCurriculum)
                  Container(
                    margin: const EdgeInsets.only(bottom: 4),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.amber.shade100,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(Icons.info_outline,
                          size: 12, color: Colors.amber.shade800),
                      const SizedBox(width: 4),
                      Text(
                        'Off-curriculum',
                        style: TextStyle(
                            fontSize: 10, color: Colors.amber.shade800),
                      ),
                    ]),
                  ),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    color: isUser
                        ? const Color(0xFF3B82F6)
                        : message.isOffCurriculum
                            ? Colors.amber.shade50
                            : cs.surface,
                    borderRadius: BorderRadius.only(
                      topLeft: const Radius.circular(16),
                      topRight: const Radius.circular(16),
                      bottomLeft: Radius.circular(isUser ? 16 : 4),
                      bottomRight: Radius.circular(isUser ? 4 : 16),
                    ),
                    border: isUser
                        ? null
                        : Border.all(
                            color: message.isOffCurriculum
                                ? Colors.amber.shade200
                                : cs.outlineVariant),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 6,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Text(
                    message.text,
                    style: TextStyle(
                      fontSize: 14,
                      height: 1.5,
                      color: isUser ? Colors.white : cs.onSurface,
                    ),
                    textDirection: TextDirection.rtl,
                  ),
                ),
                if (message.sources.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 4,
                    runSpacing: 4,
                    children: message.sources
                        .where((s) => s['source'] != null)
                        .map((s) => _SourceTag(source: s))
                        .toList(),
                  ),
                ],
              ],
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 14,
              backgroundColor: const Color(0xFF3B82F6).withValues(alpha: 0.15),
              child: const Icon(Icons.person_rounded,
                  size: 16, color: Color(0xFF3B82F6)),
            ),
          ],
        ],
      ),
    );
  }
}

class _SourceTag extends StatelessWidget {
  final Map<String, dynamic> source;
  const _SourceTag({required this.source});

  @override
  Widget build(BuildContext context) {
    final grade = source['grade'] ?? '';
    final page = source['page'];
    final label =
        'Grade $grade${page != null ? ' · p.$page' : ''}';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
      decoration: BoxDecoration(
        color: const Color(0xFF6366F1).withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
            color: const Color(0xFF6366F1).withValues(alpha: 0.25)),
      ),
      child: Text(
        label,
        style: const TextStyle(
            fontSize: 10,
            color: Color(0xFF6366F1),
            fontWeight: FontWeight.w500),
      ),
    );
  }
}

// ── Typing indicator ──────────────────────────────────────────────────────────

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 900))
        ..repeat(reverse: true);

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            width: 28,
            height: 28,
            decoration: const BoxDecoration(
              gradient: LinearGradient(colors: _kGradient),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.auto_awesome_rounded,
                color: Colors.white, size: 14),
          ),
          const SizedBox(width: 8),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: cs.surface,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(16),
                topRight: Radius.circular(16),
                bottomRight: Radius.circular(16),
                bottomLeft: Radius.circular(4),
              ),
              border: Border.all(color: cs.outlineVariant),
            ),
            child: AnimatedBuilder(
              animation: _ctrl,
              builder: (context, _) => Row(
                mainAxisSize: MainAxisSize.min,
                children: List.generate(3, (i) {
                  final delay = i * 0.3;
                  final t = (_ctrl.value - delay).clamp(0.0, 1.0);
                  return Container(
                    width: 6,
                    height: 6,
                    margin: const EdgeInsets.symmetric(horizontal: 2),
                    decoration: BoxDecoration(
                      color: Color.lerp(
                        cs.onSurfaceVariant.withValues(alpha: 0.3),
                        const Color(0xFF3B82F6),
                        t,
                      ),
                      shape: BoxShape.circle,
                    ),
                  );
                }),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Error banner ──────────────────────────────────────────────────────────────

class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner(this.message);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        borderRadius: Radii.mdRadius,
        border: Border.all(color: Colors.red.shade200),
      ),
      child: Row(children: [
        Icon(Icons.error_outline, color: Colors.red.shade600, size: 18),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            message,
            style: TextStyle(
                fontSize: 13,
                color: Colors.red.shade700,
                height: 1.4),
            textDirection: TextDirection.rtl,
          ),
        ),
      ]),
    );
  }
}

// ── Input bar ─────────────────────────────────────────────────────────────────

class _InputBar extends StatelessWidget {
  final TextEditingController controller;
  final VoidCallback onSend;

  const _InputBar({required this.controller, required this.onSend});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final bottom = MediaQuery.of(context).padding.bottom;
    final isSending =
        context.watch<ChatbotCubit>().state is ChatbotSending;

    return Container(
      padding: EdgeInsets.fromLTRB(16, 10, 16, bottom + 10),
      decoration: BoxDecoration(
        color: cs.surface,
        border: Border(top: BorderSide(color: cs.outlineVariant)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(children: [
        Expanded(
          child: TextField(
            controller: controller,
            textDirection: TextDirection.rtl,
            minLines: 1,
            maxLines: 4,
            enabled: !isSending,
            onSubmitted: (_) => onSend(),
            decoration: InputDecoration(
              hintText: 'اكتب سؤالك هنا...',
              hintStyle: TextStyle(color: cs.onSurfaceVariant, fontSize: 14),
              hintTextDirection: TextDirection.rtl,
              contentPadding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              filled: true,
              fillColor: cs.surfaceContainerLowest,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: BorderSide(color: cs.outlineVariant),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: BorderSide(color: cs.outlineVariant),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: const BorderSide(color: Color(0xFF3B82F6), width: 1.5),
              ),
            ),
          ),
        ),
        const SizedBox(width: 8),
        AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          child: isSending
              ? Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: const Color(0xFF3B82F6).withValues(alpha: 0.12),
                    shape: BoxShape.circle,
                  ),
                  child: const Padding(
                    padding: EdgeInsets.all(10),
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Color(0xFF3B82F6),
                    ),
                  ),
                )
              : GestureDetector(
                  onTap: onSend,
                  child: Container(
                    width: 44,
                    height: 44,
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(colors: _kGradient),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.send_rounded,
                        color: Colors.white, size: 20),
                  ),
                ),
        ),
      ]),
    );
  }
}
