import 'package:dio/dio.dart';
import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

const _kAiBaseUrl = 'http://localhost:8005';
const _kApiKey = 'change-me-shared-secret';

const _kGradient = [Color(0xFF6366F1), Color(0xFF8B5CF6)];

// ─────────────────────────────────────────────────────────────────────────────

class TeacherContentRecommenderScreen extends StatefulWidget {
  const TeacherContentRecommenderScreen({super.key});

  @override
  State<TeacherContentRecommenderScreen> createState() =>
      _TeacherContentRecommenderScreenState();
}

class _TeacherContentRecommenderScreenState
    extends State<TeacherContentRecommenderScreen> {
  final _queryCtrl = TextEditingController();
  final _dio = Dio();

  final _selectedTypes = <String>{'video', 'article', 'pdf', 'image'};
  String _langPref = 'both'; // 'ar' | 'en' | 'both'

  List<_ResultItem> _results = [];
  bool _loading = false;
  String? _error;
  String? _topicAr;
  int? _searchMs;

  @override
  void dispose() {
    _queryCtrl.dispose();
    _dio.close();
    super.dispose();
  }

  Future<void> _search() async {
    final q = _queryCtrl.text.trim();
    if (q.isEmpty) return;
    setState(() {
      _loading = true;
      _error = null;
      _results = [];
    });

    try {
      final resp = await _dio.post(
        '$_kAiBaseUrl/recommend',
        options: Options(
          headers: {'X-API-Key': _kApiKey},
          receiveTimeout: const Duration(seconds: 60),
          sendTimeout: const Duration(seconds: 10),
        ),
        data: {
          'query': q,
          'content_types': _selectedTypes.isEmpty
              ? ['video', 'article', 'pdf', 'image']
              : _selectedTypes.toList(),
          'language_preference': _langPref,
          'max_results': 15,
        },
      );

      final data = resp.data as Map<String, dynamic>;
      final raw = data['results'] as List<dynamic>;
      setState(() {
        _topicAr = data['topic_ar'] as String?;
        _searchMs = data['search_time_ms'] as int?;
        _results = raw
            .map((e) => _ResultItem.fromJson(e as Map<String, dynamic>))
            .toList();
        _loading = false;
      });
    } on DioException catch (e) {
      setState(() {
        _error = e.response != null
            ? 'Server error ${e.response!.statusCode}: ${e.response!.data}'
            : 'Could not reach the AI service.\nMake sure it is running on port 8005.';
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Unexpected error: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: cs.surfaceContainerLowest,
      body: CustomScrollView(
        slivers: [
          // ── Hero ─────────────────────────────────────────────────────────
          SliverToBoxAdapter(
            child: GradientHero(
              greeting: 'Content Recommender',
              subtitle: 'Find videos, articles, PDFs & images',
              colors: _kGradient,
            ),
          ),

          // ── Search card ──────────────────────────────────────────────────
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
              child: AppCard.surface(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'What are you teaching?',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w700,
                          ),
                    ),
                    const SizedBox(height: 10),
                    Directionality(
                      textDirection: TextDirection.rtl,
                      child: TextField(
                        controller: _queryCtrl,
                        maxLines: 3,
                        minLines: 2,
                        textDirection: TextDirection.rtl,
                        style: const TextStyle(fontSize: 15, height: 1.5),
                        decoration: InputDecoration(
                          hintText:
                              'اكتب موضوع الدرس بالعربي…\nمثال: شرح الكهرباء والتيار الكهربائي',
                          hintTextDirection: TextDirection.rtl,
                          filled: true,
                          fillColor: cs.surfaceContainerHigh,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: const EdgeInsets.all(14),
                        ),
                      ),
                    ),
                    const SizedBox(height: 14),

                    // Content type selector
                    Text(
                      'Content type',
                      style: Theme.of(context).textTheme.labelMedium?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: [
                        _TypeChip(
                          label: 'Video',
                          icon: Icons.play_circle_rounded,
                          color: const Color(0xFFEF4444),
                          selected: _selectedTypes.contains('video'),
                          onTap: () => setState(() => _selectedTypes.contains('video')
                              ? _selectedTypes.remove('video')
                              : _selectedTypes.add('video')),
                        ),
                        _TypeChip(
                          label: 'Article',
                          icon: Icons.article_rounded,
                          color: const Color(0xFF6366F1),
                          selected: _selectedTypes.contains('article'),
                          onTap: () => setState(() => _selectedTypes.contains('article')
                              ? _selectedTypes.remove('article')
                              : _selectedTypes.add('article')),
                        ),
                        _TypeChip(
                          label: 'PDF',
                          icon: Icons.picture_as_pdf_rounded,
                          color: const Color(0xFFF59E0B),
                          selected: _selectedTypes.contains('pdf'),
                          onTap: () => setState(() => _selectedTypes.contains('pdf')
                              ? _selectedTypes.remove('pdf')
                              : _selectedTypes.add('pdf')),
                        ),
                        _TypeChip(
                          label: 'Image',
                          icon: Icons.image_rounded,
                          color: const Color(0xFF10B981),
                          selected: _selectedTypes.contains('image'),
                          onTap: () => setState(() => _selectedTypes.contains('image')
                              ? _selectedTypes.remove('image')
                              : _selectedTypes.add('image')),
                        ),
                      ],
                    ),
                    const SizedBox(height: 14),

                    // Language preference
                    Text(
                      'Language',
                      style: Theme.of(context).textTheme.labelMedium?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                    ),
                    const SizedBox(height: 8),
                    SegmentedButton<String>(
                      segments: const [
                        ButtonSegment(
                          value: 'ar',
                          label: Text('العربية'),
                          icon: Icon(Icons.language_rounded, size: 16),
                        ),
                        ButtonSegment(
                          value: 'both',
                          label: Text('Both'),
                          icon: Icon(Icons.public_rounded, size: 16),
                        ),
                        ButtonSegment(
                          value: 'en',
                          label: Text('English'),
                          icon: Icon(Icons.language_rounded, size: 16),
                        ),
                      ],
                      selected: {_langPref},
                      onSelectionChanged: (v) =>
                          setState(() => _langPref = v.first),
                    ),
                    const SizedBox(height: 14),

                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _loading ? null : _search,
                        icon: _loading
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Icon(Icons.search_rounded),
                        label: Text(_loading ? 'Searching…' : 'Find Content'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ── Error ────────────────────────────────────────────────────────
          if (_error != null)
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                child: AppCard.filled(
                  color: Theme.of(context).colorScheme.errorContainer,
                  padding: const EdgeInsets.all(14),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.error_outline_rounded,
                          color: cs.onErrorContainer, size: 20),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          _error!,
                          style: TextStyle(
                              color: cs.onErrorContainer, fontSize: 13),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

          // ── Stats bar ────────────────────────────────────────────────────
          if (_results.isNotEmpty)
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
                child: Row(
                  children: [
                    if (_topicAr != null) ...[
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFF6366F1).withValues(alpha: 0.12),
                          borderRadius: Radii.pillRadius,
                        ),
                        child: Text(
                          _topicAr!,
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFF6366F1),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                    ],
                    Text(
                      '${_results.length} results',
                      style:
                          TextStyle(fontSize: 13, color: cs.onSurfaceVariant),
                    ),
                    if (_searchMs != null) ...[
                      const SizedBox(width: 6),
                      Text(
                        '· ${(_searchMs! / 1000).toStringAsFixed(1)}s',
                        style: TextStyle(
                            fontSize: 12, color: cs.onSurfaceVariant),
                      ),
                    ],
                  ],
                ),
              ),
            ),

          // ── Results list ─────────────────────────────────────────────────
          SliverList(
            delegate: SliverChildBuilderDelegate(
              (_, i) => Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
                child: _ResultCard(item: _results[i]),
              ),
              childCount: _results.length,
            ),
          ),

          // ── Bottom spacer ─────────────────────────────────────────────────
          SliverToBoxAdapter(
            child: SizedBox(
                height: MediaQuery.of(context).padding.bottom + 24),
          ),
        ],
      ),
    );
  }

}

// ── Data model ────────────────────────────────────────────────────────────────

class _ResultItem {
  final String type;
  final String title;
  final String url;
  final String description;
  final String source;
  final String language;
  final double score;
  final String? duration;
  final int? year;

  const _ResultItem({
    required this.type,
    required this.title,
    required this.url,
    required this.description,
    required this.source,
    required this.language,
    required this.score,
    this.duration,
    this.year,
  });

  factory _ResultItem.fromJson(Map<String, dynamic> j) => _ResultItem(
        type: j['content_type'] as String? ?? 'article',
        title: j['title'] as String? ?? '',
        url: j['url'] as String? ?? '',
        description: j['description'] as String? ?? '',
        source: j['source'] as String? ?? '',
        language: j['language'] as String? ?? '',
        score: ((j['relevance_score'] as num?) ?? 0).toDouble(),
        duration: j['duration'] as String?,
        year: j['year'] as int?,
      );
}

// ── Result card ───────────────────────────────────────────────────────────────

class _ResultCard extends StatelessWidget {
  final _ResultItem item;
  const _ResultCard({required this.item});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final typeColor = _colorFor(item.type);

    return AppCard.surface(
      padding: const EdgeInsets.all(14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Type icon
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: typeColor.withValues(alpha: 0.12),
              borderRadius: Radii.smRadius,
            ),
            child: Icon(_iconFor(item.type), color: typeColor, size: 20),
          ),
          const SizedBox(width: 12),

          // Text content
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.title,
                  style: const TextStyle(
                      fontWeight: FontWeight.w700, fontSize: 13.5),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                if (item.description.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(
                    item.description,
                    style: TextStyle(
                        fontSize: 12,
                        color: cs.onSurfaceVariant,
                        height: 1.4),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
                const SizedBox(height: 8),
                Wrap(
                  spacing: 6,
                  runSpacing: 4,
                  children: [
                    if (item.source.isNotEmpty)
                      _Tag(item.source, cs.onSurfaceVariant,
                          cs.surfaceContainerHigh),
                    if (item.language.isNotEmpty &&
                        item.language != 'unknown')
                      _Tag(item.language.toUpperCase(),
                          const Color(0xFF0891B2),
                          const Color(0xFF0891B2).withValues(alpha: 0.10)),
                    if (item.duration != null)
                      _Tag(item.duration!, typeColor,
                          typeColor.withValues(alpha: 0.10)),
                    if (item.year != null)
                      _Tag('${item.year}', cs.onSurfaceVariant,
                          cs.surfaceContainerHigh),
                    _Tag(
                      '${(item.score * 100).round()}%',
                      _scoreColor(item.score),
                      _scoreColor(item.score).withValues(alpha: 0.12),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Open button
          const SizedBox(width: 8),
          IconButton(
            onPressed: () => _open(item.url),
            icon: const Icon(Icons.open_in_new_rounded, size: 20),
            color: typeColor,
            tooltip: 'Open',
            style: IconButton.styleFrom(
              backgroundColor: typeColor.withValues(alpha: 0.10),
              shape: RoundedRectangleBorder(
                  borderRadius: Radii.smRadius),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _open(String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null) return;
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      await launchUrl(uri, mode: LaunchMode.platformDefault);
    }
  }

  Color _colorFor(String type) {
    switch (type) {
      case 'video':   return const Color(0xFFEF4444);
      case 'pdf':     return const Color(0xFFF59E0B);
      case 'image':   return const Color(0xFF10B981);
      default:        return const Color(0xFF6366F1);
    }
  }

  IconData _iconFor(String type) {
    switch (type) {
      case 'video':   return Icons.play_circle_rounded;
      case 'pdf':     return Icons.picture_as_pdf_rounded;
      case 'image':   return Icons.image_rounded;
      default:        return Icons.article_rounded;
    }
  }

  Color _scoreColor(double score) {
    if (score >= 0.80) return const Color(0xFF10B981);
    if (score >= 0.60) return const Color(0xFFF59E0B);
    return const Color(0xFF6B7280);
  }
}

// ── Type chip ─────────────────────────────────────────────────────────────────

class _TypeChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final bool selected;
  final VoidCallback onTap;
  const _TypeChip({
    required this.label,
    required this.icon,
    required this.color,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? color.withValues(alpha: 0.13) : cs.surfaceContainerHigh,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: selected ? color : Colors.transparent,
            width: 1.5,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 15,
                color: selected ? color : cs.onSurfaceVariant),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 13,
                fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                color: selected ? color : cs.onSurfaceVariant,
              ),
            ),
            if (selected) ...[
              const SizedBox(width: 4),
              Icon(Icons.check_circle_rounded, size: 13, color: color),
            ],
          ],
        ),
      ),
    );
  }
}

// ── Tag chip ──────────────────────────────────────────────────────────────────

class _Tag extends StatelessWidget {
  final String label;
  final Color color;
  final Color bg;
  const _Tag(this.label, this.color, this.bg);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(label,
          style: TextStyle(
              fontSize: 11,
              color: color,
              fontWeight: FontWeight.w600)),
    );
  }
}
