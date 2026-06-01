import 'package:dio/dio.dart';
import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

const _kAiBaseUrl = 'http://10.0.2.2:8005'; // Android emulator → host
// For web/Chrome testing use: 'http://localhost:8005'
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

  final _allTypes = ['video', 'article', 'pdf', 'image'];
  final _selectedTypes = <String>{'video', 'article', 'pdf', 'image'};

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
    if (_selectedTypes.isEmpty) {
      setState(() => _error = 'Please select at least one content type.');
      return;
    }

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
          'content_types': _selectedTypes.toList(),
          'language_preference': 'both',
          'max_results': 15,
        },
      );

      final data = resp.data as Map<String, dynamic>;
      final raw = data['results'] as List<dynamic>;
      setState(() {
        _topicAr = data['topic_ar'] as String?;
        _searchMs = data['search_time_ms'] as int?;
        _results = raw.map((e) => _ResultItem.fromJson(e as Map<String, dynamic>)).toList();
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

          // ── Search bar ───────────────────────────────────────────────────
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

                    // Content type chips
                    Text(
                      'Content types',
                      style: Theme.of(context).textTheme.labelMedium?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      children: _allTypes.map((t) {
                        final selected = _selectedTypes.contains(t);
                        return FilterChip(
                          label: Text(_typeLabel(t)),
                          avatar: Icon(_typeIcon(t), size: 14),
                          selected: selected,
                          onSelected: (v) {
                            setState(() {
                              if (v) {
                                _selectedTypes.add(t);
                              } else {
                                _selectedTypes.remove(t);
                              }
                            });
                          },
                        );
                      }).toList(),
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
                  color: cs.errorContainer,
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
                      style: TextStyle(
                          fontSize: 13, color: cs.onSurfaceVariant),
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

          // ── Results ──────────────────────────────────────────────────────
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

  String _typeLabel(String t) {
    switch (t) {
      case 'video':   return 'Video';
      case 'article': return 'Article';
      case 'pdf':     return 'PDF';
      case 'image':   return 'Image';
      default:        return t;
    }
  }

  IconData _typeIcon(String t) {
    switch (t) {
      case 'video':   return Icons.play_circle_outline_rounded;
      case 'article': return Icons.article_outlined;
      case 'pdf':     return Icons.picture_as_pdf_outlined;
      case 'image':   return Icons.image_outlined;
      default:        return Icons.link_rounded;
    }
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
  final String? thumbnail;
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
    this.thumbnail,
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
        thumbnail: j['thumbnail_url'] as String?,
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

          // Content
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Title
                Text(
                  item.title,
                  style: const TextStyle(
                      fontWeight: FontWeight.w700, fontSize: 13.5),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 4),

                // Description
                if (item.description.isNotEmpty)
                  Text(
                    item.description,
                    style: TextStyle(
                        fontSize: 12, color: cs.onSurfaceVariant, height: 1.4),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),

                const SizedBox(height: 8),

                // Metadata row
                Wrap(
                  spacing: 6,
                  runSpacing: 4,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    // Source
                    if (item.source.isNotEmpty)
                      _Chip(
                        label: item.source,
                        color: cs.onSurfaceVariant,
                        bg: cs.surfaceContainerHigh,
                      ),
                    // Language
                    if (item.language.isNotEmpty && item.language != 'unknown')
                      _Chip(
                        label: item.language.toUpperCase(),
                        color: const Color(0xFF0891B2),
                        bg: const Color(0xFF0891B2).withValues(alpha: 0.10),
                      ),
                    // Duration (video)
                    if (item.duration != null)
                      _Chip(
                        label: item.duration!,
                        color: typeColor,
                        bg: typeColor.withValues(alpha: 0.10),
                      ),
                    // Year
                    if (item.year != null)
                      _Chip(
                        label: '${item.year}',
                        color: cs.onSurfaceVariant,
                        bg: cs.surfaceContainerHigh,
                      ),
                    // Score badge
                    _Chip(
                      label: '${(item.score * 100).round()}%',
                      color: _scoreColor(item.score),
                      bg: _scoreColor(item.score).withValues(alpha: 0.12),
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
                borderRadius: Radii.smRadius,
              ),
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
      case 'video':   return const Color(0xFFEF4444); // red
      case 'pdf':     return const Color(0xFFF59E0B); // amber
      case 'image':   return const Color(0xFF10B981); // emerald
      default:        return const Color(0xFF6366F1); // indigo
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

class _Chip extends StatelessWidget {
  final String label;
  final Color color;
  final Color bg;
  const _Chip({required this.label, required this.color, required this.bg});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }
}
