import 'package:first_try/core/theme/app_colors.dart';
import 'package:flutter/material.dart';

/// A single stat shown inside the [GradientHero] stat strip.
class HeroStat {
  final String value;
  final String label;
  const HeroStat({required this.value, required this.label});
}

/// Full-bleed gradient hero block. Designed for the top of home screens —
/// holds a greeting, subtitle, optional stats strip, and an optional trailing slot.
class GradientHero extends StatefulWidget {
  final String greeting;
  final String? subtitle;
  final Widget? trailing;
  final List<Color>? colors;
  final List<HeroStat>? stats;
  final EdgeInsetsGeometry padding;
  final BorderRadius radius;

  const GradientHero({
    super.key,
    required this.greeting,
    this.subtitle,
    this.trailing,
    this.colors,
    this.stats,
    this.padding = const EdgeInsets.fromLTRB(20, 24, 20, 24),
    this.radius = const BorderRadius.only(
      bottomLeft: Radius.circular(28),
      bottomRight: Radius.circular(28),
    ),
  });

  @override
  State<GradientHero> createState() => _GradientHeroState();
}

class _GradientHeroState extends State<GradientHero> {
  bool _showBlob = false;

  @override
  Widget build(BuildContext context) {
    final palette = context.palette;
    final gradColors = widget.colors ?? palette.brandGradient;

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: gradColors,
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: widget.radius,
        boxShadow: [
          BoxShadow(
            color: gradColors.first.withValues(alpha: 0.28),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Stack(
        children: [
          // Decorative glow blob (top-right) — only visible while hovering the trailing slot
          Positioned(
            top: -50,
            right: -40,
            child: IgnorePointer(
              child: AnimatedOpacity(
                opacity: _showBlob ? 1.0 : 0.0,
                duration: const Duration(milliseconds: 220),
                curve: Curves.easeOut,
                child: Container(
                  width: 180,
                  height: 180,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white.withValues(alpha: 0.10),
                  ),
                ),
              ),
            ),
          ),
          SafeArea(
            bottom: false,
            child: Padding(
              padding: widget.padding,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (widget.subtitle != null)
                              Text(
                                widget.subtitle!,
                                style: Theme.of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(
                                      color:
                                          Colors.white.withValues(alpha: 0.80),
                                      fontWeight: FontWeight.w500,
                                    ),
                              ),
                            if (widget.subtitle != null) const SizedBox(height: 4),
                            Text(
                              widget.greeting,
                              style: Theme.of(context)
                                  .textTheme
                                  .headlineMedium
                                  ?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w800,
                                    letterSpacing: -0.4,
                                  ),
                            ),
                          ],
                        ),
                      ),
                      if (widget.trailing != null) ...[
                        const SizedBox(width: 16),
                        MouseRegion(
                          onEnter: (_) => setState(() => _showBlob = true),
                          onExit: (_) => setState(() => _showBlob = false),
                          child: widget.trailing!,
                        ),
                      ],
                    ],
                  ),

                  // ── Stat strip ──────────────────────────────────────────
                  if (widget.stats != null && widget.stats!.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        for (int i = 0; i < widget.stats!.length; i++) ...[
                          if (i > 0) const SizedBox(width: 10),
                          Expanded(child: _StatChip(stat: widget.stats![i])),
                        ],
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatChip extends StatelessWidget {
  final HeroStat stat;
  const _StatChip({required this.stat});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.15),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            stat.value,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w800,
              color: Colors.white,
              height: 1.0,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
          const SizedBox(height: 3),
          Text(
            stat.label,
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w500,
              color: Colors.white.withValues(alpha: 0.80),
              height: 1.2,
            ),
          ),
        ],
      ),
    );
  }
}

/// Section title used between sliver / column groups.
class SectionHeader extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final EdgeInsetsGeometry padding;

  const SectionHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.trailing,
    this.padding = const EdgeInsets.fromLTRB(20, 20, 20, 8),
  });

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: padding,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  title,
                  style: tt.titleMedium?.copyWith(fontWeight: FontWeight.w800),
                ),
                if (subtitle != null)
                  Text(
                    subtitle!,
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                  ),
              ],
            ),
          ),
          ?trailing,
        ],
      ),
    );
  }
}
