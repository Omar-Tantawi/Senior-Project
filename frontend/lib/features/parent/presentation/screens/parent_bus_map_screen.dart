import 'dart:async';

import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/features/parent/data/models/parent_models.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_cubit.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:intl/intl.dart';
import 'package:latlong2/latlong.dart';

/// Fullscreen, fully-interactive bus map. Pushed when the parent taps the
/// inline preview on [ParentBusScreen]. Live-pings refresh every 15 s.
class ParentBusMapScreen extends StatefulWidget {
  final int childId;
  const ParentBusMapScreen({super.key, required this.childId});

  @override
  State<ParentBusMapScreen> createState() => _ParentBusMapScreenState();
}

class _ParentBusMapScreenState extends State<ParentBusMapScreen>
    with SingleTickerProviderStateMixin {
  static const _pollEvery = Duration(seconds: 5);
  static const _tweenDuration = Duration(milliseconds: 4500);

  final _mapController = MapController();
  Timer? _timer;
  bool _followBus = true;
  double _zoom = 16;

  late final AnimationController _tween;
  LatLng? _fromPoint;
  LatLng? _toPoint;

  LatLng? get _animatedPoint {
    if (_fromPoint == null || _toPoint == null) return null;
    final t = Curves.easeInOut.transform(_tween.value);
    return LatLng(
      _fromPoint!.latitude + (_toPoint!.latitude - _fromPoint!.latitude) * t,
      _fromPoint!.longitude + (_toPoint!.longitude - _fromPoint!.longitude) * t,
    );
  }

  void _ingestPoint(LatLng next) {
    if (_toPoint != null &&
        next.latitude == _toPoint!.latitude &&
        next.longitude == _toPoint!.longitude) {
      return;
    }
    _fromPoint = _animatedPoint ?? next;
    _toPoint = next;
    _tween
      ..reset()
      ..forward();
  }

  @override
  void initState() {
    super.initState();
    _tween = AnimationController(vsync: this, duration: _tweenDuration)
      ..addListener(() => setState(() {}));
    _timer = Timer.periodic(_pollEvery, (_) {
      if (mounted) context.read<ParentCubit>().refreshBusLive();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _tween.dispose();
    _mapController.dispose();
    super.dispose();
  }

  void _recenter(LatLng point) {
    _mapController.move(point, _zoom);
    setState(() => _followBus = true);
  }

  void _zoomIn(LatLng point) {
    _zoom = (_zoom + 1).clamp(3, 19);
    _mapController.move(point, _zoom);
  }

  void _zoomOut(LatLng point) {
    _zoom = (_zoom - 1).clamp(3, 19);
    _mapController.move(point, _zoom);
  }

  List<Marker> _markers(ParentBusModel bus, LatLng busPoint) {
    final out = <Marker>[];
    final school = bus.schoolStop;
    for (final s in bus.stops) {
      if (!s.hasLocation) continue;
      final isSchool = school != null && s.id == school.id;
      final isPickup = bus.pickupStopId != null && s.id == bus.pickupStopId;
      if (isSchool) continue;
      out.add(Marker(
        point: LatLng(s.latitude!, s.longitude!),
        width: 36,
        height: 36,
        child: _StopDot(highlight: isPickup, label: s.name),
      ));
    }
    // When the bus parks at the school the two markers stack. Nudge the bus
    // slightly west (in lat/lng) so both pins remain visible side-by-side.
    final atSchool = school != null &&
        school.hasLocation &&
        (busPoint.latitude - school.latitude!).abs() < 0.0003 &&
        (busPoint.longitude - school.longitude!).abs() < 0.0003;
    const lngOffset = 0.00025; // ~25 m at this latitude
    final busDrawPoint = atSchool
        ? LatLng(busPoint.latitude, busPoint.longitude - lngOffset)
        : busPoint;

    out.add(Marker(
      point: busDrawPoint,
      width: 56,
      height: 56,
      child: const _BusMarker(),
    ));
    if (school != null && school.hasLocation) {
      out.add(Marker(
        point: LatLng(school.latitude!, school.longitude!),
        width: 52,
        height: 52,
        child: const _SchoolMarker(),
      ));
    }
    return out;
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<ParentCubit, ParentState>(
      builder: (context, state) {
        if (state is! ParentLoaded) return const _EmptyScaffold();
        final bus = state.bus[widget.childId];
        if (bus == null || !bus.hasLocation) {
          return const _EmptyScaffold(message: 'Live location unavailable.');
        }

        final rawPoint = LatLng(bus.latitude!, bus.longitude!);
        // Hand the latest ping to the tween — animated point is what we draw.
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) _ingestPoint(rawPoint);
        });
        final busPoint = _animatedPoint ?? rawPoint;

        // Auto-follow the bus along the smooth animation.
        if (_followBus) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (mounted) _mapController.move(busPoint, _zoom);
          });
        }

        return Scaffold(
          body: Stack(
            children: [
              // Map
              FlutterMap(
                mapController: _mapController,
                options: MapOptions(
                  initialCenter: busPoint,
                  initialZoom: _zoom,
                  minZoom: 3,
                  maxZoom: 19,
                  interactionOptions: const InteractionOptions(
                    flags: InteractiveFlag.all,
                  ),
                  onPositionChanged: (pos, hasGesture) {
                    if (hasGesture && _followBus) {
                      setState(() => _followBus = false);
                    }
                    if (pos.zoom != _zoom) _zoom = pos.zoom;
                  },
                ),
                children: [
                  TileLayer(
                    urlTemplate:
                        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                    userAgentPackageName: 'com.school.app',
                    maxZoom: 19,
                  ),
                  MarkerLayer(markers: _markers(bus, busPoint)),
                ],
              ),

              // Top overlay: close button + bus info card
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _CircleBtn(
                        icon: Icons.close_rounded,
                        onTap: () => Navigator.of(context).pop(),
                      ),
                      const SizedBox(width: 10),
                      Expanded(child: _InfoCard(bus: bus)),
                    ],
                  ),
                ),
              ),

              // Bottom-left legend
              const Positioned(
                left: 12,
                bottom: 32,
                child: _Legend(),
              ),

              // Right-side controls
              Positioned(
                right: 12,
                bottom: 32,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _CircleBtn(
                      icon: Icons.add_rounded,
                      onTap: () => _zoomIn(busPoint),
                    ),
                    const SizedBox(height: 10),
                    _CircleBtn(
                      icon: Icons.remove_rounded,
                      onTap: () => _zoomOut(busPoint),
                    ),
                    const SizedBox(height: 16),
                    _CircleBtn(
                      icon: _followBus
                          ? Icons.my_location_rounded
                          : Icons.location_searching_rounded,
                      highlight: _followBus,
                      onTap: () => _recenter(busPoint),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

// ── Bus info overlay card ─────────────────────────────────────────────────────

class _InfoCard extends StatelessWidget {
  final ParentBusModel bus;
  const _InfoCard({required this.bus});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: cs.surface.withValues(alpha: 0.94),
        borderRadius: Radii.mdRadius,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Icon(Icons.directions_bus_rounded,
                  color: Color(0xFF6366F1), size: 18),
              const SizedBox(width: 6),
              Text(
                bus.busPlate,
                style: const TextStyle(
                    fontWeight: FontWeight.w800, letterSpacing: 1.1),
              ),
              const Spacer(),
              Container(
                width: 7,
                height: 7,
                decoration: const BoxDecoration(
                  color: Color(0xFF10B981),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 5),
              const Text('Live',
                  style: TextStyle(
                      fontSize: 11,
                      color: Color(0xFF10B981),
                      fontWeight: FontWeight.w700)),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '${bus.latitude!.toStringAsFixed(5)}, ${bus.longitude!.toStringAsFixed(5)}',
            style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
          ),
          if (bus.updatedAt != null)
            Text(
              'Updated ${_rel(bus.updatedAt!)}',
              style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
            ),
        ],
      ),
    );
  }

  String _rel(String iso) {
    try {
      final d = DateTime.parse(iso);
      final diff = DateTime.now().difference(d);
      if (diff.inSeconds < 60) return 'just now';
      if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
      if (diff.inHours < 24) return '${diff.inHours}h ago';
      return DateFormat('d MMM, h:mm a').format(d);
    } catch (_) {
      return '';
    }
  }
}

// ── Legend ────────────────────────────────────────────────────────────────────

class _Legend extends StatelessWidget {
  const _Legend();

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: cs.surface.withValues(alpha: 0.94),
        borderRadius: Radii.smRadius,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: const Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _LegendRow(color: Color(0xFF6366F1), label: 'Bus'),
          SizedBox(height: 4),
          _LegendRow(color: Color(0xFF10B981), label: 'Your stop'),
          SizedBox(height: 4),
          _LegendRow(color: Color(0xFF94A3B8), label: 'Other stops'),
          SizedBox(height: 4),
          _LegendRow(color: Color(0xFFF59E0B), label: 'School'),
        ],
      ),
    );
  }
}

class _LegendRow extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendRow({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Container(
        width: 10,
        height: 10,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      ),
      const SizedBox(width: 6),
      Text(label, style: const TextStyle(fontSize: 11)),
    ]);
  }
}

// ── Round button used for close / zoom / recenter ─────────────────────────────

class _CircleBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final bool highlight;

  const _CircleBtn({
    required this.icon,
    required this.onTap,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      color: highlight ? const Color(0xFF6366F1) : cs.surface,
      shape: const CircleBorder(),
      elevation: 4,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: SizedBox(
          width: 44,
          height: 44,
          child: Icon(
            icon,
            color: highlight ? Colors.white : cs.onSurface,
            size: 22,
          ),
        ),
      ),
    );
  }
}

// ── Stop dot (pickup highlighted) ─────────────────────────────────────────────

class _StopDot extends StatelessWidget {
  final bool highlight;
  final String label;
  const _StopDot({required this.highlight, required this.label});

  @override
  Widget build(BuildContext context) {
    final color =
        highlight ? const Color(0xFF10B981) : const Color(0xFF94A3B8);
    return Tooltip(
      message: label,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          shape: BoxShape.circle,
          border: Border.all(color: color, width: 3),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.15),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Center(
          child: Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
        ),
      ),
    );
  }
}

// ── School marker ─────────────────────────────────────────────────────────────

class _SchoolMarker extends StatelessWidget {
  const _SchoolMarker();

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFFF59E0B),
        shape: BoxShape.circle,
        border: Border.all(color: Colors.white, width: 2),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFF59E0B).withValues(alpha: 0.45),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: const Icon(Icons.school_rounded, color: Colors.white, size: 22),
    );
  }
}

// ── Pulsing bus marker (matches inline preview) ───────────────────────────────

class _BusMarker extends StatefulWidget {
  const _BusMarker();
  @override
  State<_BusMarker> createState() => _BusMarkerState();
}

class _BusMarkerState extends State<_BusMarker>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;
  late final Animation<double> _scale;
  late final Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
    _scale = Tween<double>(begin: 1.0, end: 2.4).animate(
      CurvedAnimation(parent: _pulse, curve: Curves.easeOut),
    );
    _opacity = Tween<double>(begin: 0.55, end: 0.0).animate(
      CurvedAnimation(parent: _pulse, curve: Curves.easeOut),
    );
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        AnimatedBuilder(
          animation: _pulse,
          builder: (_, __) => Transform.scale(
            scale: _scale.value,
            child: Container(
              width: 26,
              height: 26,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF6366F1)
                    .withValues(alpha: _opacity.value),
              ),
            ),
          ),
        ),
        Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: const Color(0xFF6366F1),
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF6366F1).withValues(alpha: 0.45),
                blurRadius: 12,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: const Icon(
            Icons.directions_bus_rounded,
            color: Colors.white,
            size: 24,
          ),
        ),
      ],
    );
  }
}

// ── Empty state ───────────────────────────────────────────────────────────────

class _EmptyScaffold extends StatelessWidget {
  final String message;
  const _EmptyScaffold({this.message = 'No bus data.'});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: Center(child: Text(message)),
    );
  }
}
