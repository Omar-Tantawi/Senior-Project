import 'dart:async';

import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:first_try/features/parent/data/models/parent_models.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_cubit.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_state.dart';
import 'package:first_try/features/parent/presentation/screens/parent_bus_map_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:intl/intl.dart';
import 'package:latlong2/latlong.dart';

class ParentBusScreen extends StatelessWidget {
  const ParentBusScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Bus Tracking',
            style: TextStyle(fontWeight: FontWeight.w700)),
      ),
      body: BlocBuilder<ParentCubit, ParentState>(
        builder: (context, state) {
          if (state is! ParentLoaded) return const SizedBox.shrink();

          final busData = state.bus[state.selectedChildId];
          final childName = state.selectedChild.name.split(' ').first;

          return RefreshIndicator(
            onRefresh: () => context.read<ParentCubit>().load(),
            child: CustomScrollView(
              slivers: [
                // Child selector
                SliverToBoxAdapter(
                  child: _ChildTabBar(
                    children: state.profile.children,
                    selectedIndex: state.selectedChildIndex,
                    onSelect: (i) =>
                        context.read<ParentCubit>().selectChild(i),
                  ),
                ),

                if (busData == null) ...[
                  SliverFillRemaining(
                    child: Center(
                      child: Text(
                        'No bus assigned for $childName.',
                        style: TextStyle(
                            color: Theme.of(context)
                                .colorScheme
                                .onSurfaceVariant),
                      ),
                    ),
                  ),
                ] else ...[
                  SliverToBoxAdapter(
                      child: _BusStatusCard(bus: busData)),
                  SliverToBoxAdapter(
                      child: _BoardingStatusPill(
                          status: busData.status, childName: childName)),
                  SliverToBoxAdapter(
                      child: _BusActivityPill(activity: busData.activity)),
                  SliverToBoxAdapter(
                      child: _BusInfoSection(
                          bus: busData, childName: childName)),
                  SliverToBoxAdapter(
                      child: _BusLiveMap(
                          bus: busData,
                          childId: state.selectedChildId)),
                  const SliverToBoxAdapter(child: SizedBox(height: 24)),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}

// ── Child tab bar ─────────────────────────────────────────────────────────────

class _ChildTabBar extends StatelessWidget {
  final List<ChildSummaryModel> children;
  final int selectedIndex;
  final void Function(int) onSelect;

  const _ChildTabBar({
    required this.children,
    required this.selectedIndex,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: Row(
        children: List.generate(children.length, (i) {
          final child = children[i];
          final selected = i == selectedIndex;
          return Expanded(
            child: Padding(
              padding:
                  EdgeInsets.only(right: i < children.length - 1 ? 8 : 0),
              child: FilterPill(
                label: child.name.split(' ').first,
                selected: selected,
                onSelected: (_) => onSelect(i),
              ),
            ),
          );
        }),
      ),
    );
  }
}

// ── Bus status card ───────────────────────────────────────────────────────────

class _BusStatusCard extends StatelessWidget {
  final ParentBusModel bus;
  const _BusStatusCard({required this.bus});

  @override
  Widget build(BuildContext context) {
    final palette = context.palette;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
      child: AppCard.glass(
        gradient: palette.brandGradient,
        opacity: 0.92,
        child: Row(children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.22),
              borderRadius: Radii.smRadius,
            ),
            child: const Icon(Icons.directions_bus_rounded,
                color: Colors.white, size: 30),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    bus.busPlate,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.5),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    bus.routeName,
                    style: const TextStyle(
                        color: Colors.white70, fontSize: 13),
                  ),
                ]),
          ),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFF34D399),
              borderRadius: Radii.pillRadius,
            ),
            child: Row(children: [
              Container(
                width: 7,
                height: 7,
                decoration: const BoxDecoration(
                    color: Colors.white, shape: BoxShape.circle),
              ),
              const SizedBox(width: 5),
              const Text('Active',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w600)),
            ]),
          ),
        ]),
      ),
    );
  }
}

// ── Boarding status pill ──────────────────────────────────────────────────────

class _BoardingStatusPill extends StatelessWidget {
  final BusStatus status;
  final String childName;
  const _BoardingStatusPill({required this.status, required this.childName});

  @override
  Widget build(BuildContext context) {
    final (icon, label, color) = _styleFor(status, childName);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.10),
          borderRadius: Radii.mdRadius,
          border: Border.all(color: color.withValues(alpha: 0.35)),
        ),
        child: Row(children: [
          Icon(icon, size: 20, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w700, color: color),
            ),
          ),
        ]),
      ),
    );
  }

  (IconData, String, Color) _styleFor(BusStatus s, String name) {
    switch (s) {
      case BusStatus.onBus:
        return (
          Icons.directions_bus_filled_rounded,
          '$name is on the bus',
          const Color(0xFF6366F1),
        );
      case BusStatus.droppedOff:
        return (
          Icons.school_rounded,
          '$name was dropped off at school',
          const Color(0xFF10B981),
        );
      case BusStatus.waiting:
        return (
          Icons.access_time_rounded,
          'Waiting for pickup',
          const Color(0xFFF59E0B),
        );
      case BusStatus.noTrip:
        return (
          Icons.event_busy_rounded,
          'No trip scheduled today',
          const Color(0xFF6B7280),
        );
      case BusStatus.unknown:
        return (
          Icons.help_outline_rounded,
          'Status unavailable',
          const Color(0xFF6B7280),
        );
    }
  }
}

// ── Bus activity pill (where is the bus / what is it doing) ───────────────────

class _BusActivityPill extends StatelessWidget {
  final BusActivity activity;
  const _BusActivityPill({required this.activity});

  @override
  Widget build(BuildContext context) {
    if (activity == BusActivity.unknown) {
      return const SizedBox.shrink();
    }
    final (icon, label, color) = _styleFor(activity);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.10),
          borderRadius: Radii.mdRadius,
          border: Border.all(color: color.withValues(alpha: 0.35)),
        ),
        child: Row(children: [
          Icon(icon, size: 20, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w700, color: color),
            ),
          ),
        ]),
      ),
    );
  }

  (IconData, String, Color) _styleFor(BusActivity a) {
    switch (a) {
      case BusActivity.atHome:
        return (
          Icons.home_rounded,
          'Bus is at the pickup stop',
          const Color(0xFF10B981),
        );
      case BusActivity.headingToSchool:
        return (
          Icons.north_east_rounded,
          'On the way to school',
          const Color(0xFF6366F1),
        );
      case BusActivity.atSchool:
        return (
          Icons.school_rounded,
          'Bus is at the school',
          const Color(0xFFF59E0B),
        );
      case BusActivity.headingHome:
        return (
          Icons.south_west_rounded,
          'On the way home',
          const Color(0xFF8B5CF6),
        );
      case BusActivity.unknown:
        return (
          Icons.help_outline_rounded,
          'Activity unavailable',
          const Color(0xFF6B7280),
        );
    }
  }
}

// ── Bus info section ──────────────────────────────────────────────────────────

class _BusInfoSection extends StatelessWidget {
  final ParentBusModel bus;
  final String childName;
  const _BusInfoSection({required this.bus, required this.childName});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Text('Route Details',
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.w700)),
        ),
        _InfoRow(
          icon: Icons.place_rounded,
          label: 'Pickup Stop',
          value: bus.pickupStopName,
          color: const Color(0xFF10B981),
        ),
        const SizedBox(height: 8),
        _InfoRow(
          icon: Icons.person_rounded,
          label: 'Driver',
          value: bus.driverName ?? 'N/A',
          color: const Color(0xFF3B82F6),
        ),
        const SizedBox(height: 8),
        _InfoRow(
          icon: Icons.child_care_rounded,
          label: 'Student',
          value: childName,
          color: const Color(0xFF8B5CF6),
        ),
        if (bus.updatedAt != null) ...[
          const SizedBox(height: 8),
          _InfoRow(
            icon: Icons.update_rounded,
            label: 'Last Updated',
            value: _fmtDate(bus.updatedAt!),
            color: const Color(0xFFF59E0B),
          ),
        ],
        if (bus.hasLocation) ...[
          const SizedBox(height: 8),
          _InfoRow(
            icon: Icons.my_location_rounded,
            label: 'Bus Coordinates',
            value:
                '${bus.latitude!.toStringAsFixed(5)},  ${bus.longitude!.toStringAsFixed(5)}',
            color: const Color(0xFFEF4444),
          ),
        ],
      ]),
    );
  }

  String _fmtDate(String iso) {
    try {
      return DateFormat('d MMM, h:mm a').format(DateTime.parse(iso));
    } catch (_) {
      return iso;
    }
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _InfoRow({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return AppCard.surface(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      child: Row(children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.12),
            borderRadius: Radii.smRadius,
          ),
          child: Icon(icon, color: color, size: 18),
        ),
        const SizedBox(width: 12),
        Expanded(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(label,
                style: TextStyle(
                    fontSize: 11,
                    color: Theme.of(context).colorScheme.onSurfaceVariant)),
            Text(value,
                style: const TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w600)),
          ]),
        ),
      ]),
    );
  }
}

// ── Live map ──────────────────────────────────────────────────────────────────

/// Wraps the map and drives a 15-second polling timer via [ParentCubit].
class _BusLiveMap extends StatefulWidget {
  final ParentBusModel bus;
  final int childId;
  const _BusLiveMap({required this.bus, required this.childId});

  @override
  State<_BusLiveMap> createState() => _BusLiveMapState();
}

class _BusLiveMapState extends State<_BusLiveMap>
    with SingleTickerProviderStateMixin {
  static const _pollEvery = Duration(seconds: 5);
  static const _tweenDuration = Duration(milliseconds: 4500);

  Timer? _timer;
  final _mapController = MapController();
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

  @override
  void initState() {
    super.initState();
    _tween = AnimationController(vsync: this, duration: _tweenDuration)
      ..addListener(() => setState(() {}));
    if (widget.bus.hasLocation) {
      _fromPoint = _toPoint =
          LatLng(widget.bus.latitude!, widget.bus.longitude!);
    }
    _timer = Timer.periodic(_pollEvery, (_) {
      if (mounted) context.read<ParentCubit>().refreshBusLive();
    });
  }

  @override
  void didUpdateWidget(_BusLiveMap old) {
    super.didUpdateWidget(old);
    if (!widget.bus.hasLocation) return;
    final next = LatLng(widget.bus.latitude!, widget.bus.longitude!);
    if (_toPoint != null &&
        next.latitude == _toPoint!.latitude &&
        next.longitude == _toPoint!.longitude) {
      return; // no change
    }
    _fromPoint = _animatedPoint ?? next;
    _toPoint = next;
    _tween
      ..reset()
      ..forward();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _tween.dispose();
    _mapController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Live Location',
                style: Theme.of(context)
                    .textTheme
                    .titleMedium
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const Spacer(),
              if (widget.bus.hasLocation)
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: const Color(0xFF10B981).withValues(alpha: 0.12),
                    borderRadius: Radii.pillRadius,
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 6,
                        height: 6,
                        decoration: const BoxDecoration(
                          color: Color(0xFF10B981),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 5),
                      const Text(
                        'Live',
                        style: TextStyle(
                            fontSize: 11,
                            color: Color(0xFF10B981),
                            fontWeight: FontWeight.w700),
                      ),
                    ],
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          AppCard.surface(
            padding: EdgeInsets.zero,
            child: ClipRRect(
              borderRadius: Radii.mdRadius,
              child: SizedBox(
                height: 260,
                width: double.infinity,
                child: widget.bus.hasLocation
                    ? Stack(
                        children: [
                          _MapView(
                              bus: widget.bus,
                              controller: _mapController,
                              busPoint: _animatedPoint ??
                                  LatLng(widget.bus.latitude!,
                                      widget.bus.longitude!)),
                          // Transparent tap layer — eats map gestures so the
                          // preview can't be panned/zoomed in place.
                          Positioned.fill(
                            child: Material(
                              color: Colors.transparent,
                              child: InkWell(
                                onTap: () => _openFullscreen(context),
                              ),
                            ),
                          ),
                          Positioned(
                            top: 8,
                            right: 8,
                            child: _ExpandChip(
                                onTap: () => _openFullscreen(context)),
                          ),
                        ],
                      )
                    : _NoLocation(cs: cs),
              ),
            ),
          ),
          if (widget.bus.hasLocation && widget.bus.updatedAt != null) ...[
            const SizedBox(height: 6),
            Text(
              'Updated ${_relTime(widget.bus.updatedAt!)}  •  refreshes every 15 s',
              style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
            ),
          ],
        ],
      ),
    );
  }

  void _openFullscreen(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => BlocProvider.value(
          value: context.read<ParentCubit>(),
          child: ParentBusMapScreen(childId: widget.childId),
        ),
      ),
    );
  }

  String _relTime(String iso) {
    try {
      final d = DateTime.parse(iso);
      final diff = DateTime.now().difference(d);
      if (diff.inSeconds < 60) return 'just now';
      if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
      return '${diff.inHours}h ago';
    } catch (_) {
      return '';
    }
  }
}

// ── OpenStreetMap tile view with bus marker ───────────────────────────────────

class _MapView extends StatelessWidget {
  final ParentBusModel bus;
  final MapController controller;
  final LatLng busPoint;
  const _MapView({
    required this.bus,
    required this.controller,
    required this.busPoint,
  });

  @override
  Widget build(BuildContext context) {

    return FlutterMap(
      mapController: controller,
      options: MapOptions(
        initialCenter: busPoint,
        initialZoom: 15,
        // Preview is non-interactive — tap opens fullscreen map.
        interactionOptions: const InteractionOptions(flags: InteractiveFlag.none),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.school.app',
          maxZoom: 19,
        ),
        MarkerLayer(markers: _buildMarkers(busPoint)),
      ],
    );
  }

  List<Marker> _buildMarkers(LatLng busPoint) {
    final markers = <Marker>[];
    final school = bus.schoolStop;
    for (final s in bus.stops) {
      if (!s.hasLocation) continue;
      final isSchool = school != null && s.id == school.id;
      final isPickup = bus.pickupStopId != null && s.id == bus.pickupStopId;
      if (isSchool) continue; // school rendered separately below
      markers.add(Marker(
        point: LatLng(s.latitude!, s.longitude!),
        width: 28,
        height: 28,
        child: _StopDot(highlight: isPickup),
      ));
    }
    final atSchool = school != null &&
        school.hasLocation &&
        (busPoint.latitude - school.latitude!).abs() < 0.0003 &&
        (busPoint.longitude - school.longitude!).abs() < 0.0003;
    const lngOffset = 0.00025;
    final busDrawPoint = atSchool
        ? LatLng(busPoint.latitude, busPoint.longitude - lngOffset)
        : busPoint;

    markers.add(Marker(
      point: busDrawPoint,
      width: 48,
      height: 48,
      child: _BusMarker(),
    ));
    if (school != null && school.hasLocation) {
      markers.add(Marker(
        point: LatLng(school.latitude!, school.longitude!),
        width: 42,
        height: 42,
        child: const _SchoolMarker(),
      ));
    }
    return markers;
  }
}

// ── Stop / school markers ─────────────────────────────────────────────────────

class _StopDot extends StatelessWidget {
  final bool highlight;
  const _StopDot({required this.highlight});

  @override
  Widget build(BuildContext context) {
    final color =
        highlight ? const Color(0xFF10B981) : const Color(0xFF94A3B8);
    return Container(
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
          width: 8,
          height: 8,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
      ),
    );
  }
}

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
      child: const Icon(Icons.school_rounded, color: Colors.white, size: 20),
    );
  }
}

/// Animated pulsing bus marker.
class _BusMarker extends StatefulWidget {
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
    _scale = Tween<double>(begin: 1.0, end: 2.2).animate(
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
        // Ripple ring
        AnimatedBuilder(
          animation: _pulse,
          builder: (_, __) => Transform.scale(
            scale: _scale.value,
            child: Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF6366F1)
                    .withValues(alpha: _opacity.value),
              ),
            ),
          ),
        ),
        // Solid bus pin
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: const Color(0xFF6366F1),
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF6366F1).withValues(alpha: 0.45),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: const Icon(
            Icons.directions_bus_rounded,
            color: Colors.white,
            size: 20,
          ),
        ),
      ],
    );
  }
}

// ── Expand-to-fullscreen chip ─────────────────────────────────────────────────

class _ExpandChip extends StatelessWidget {
  final VoidCallback onTap;
  const _ExpandChip({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white.withValues(alpha: 0.95),
      borderRadius: Radii.pillRadius,
      elevation: 3,
      child: InkWell(
        borderRadius: Radii.pillRadius,
        onTap: onTap,
        child: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.open_in_full_rounded, size: 14, color: Colors.black87),
              SizedBox(width: 5),
              Text('Expand',
                  style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: Colors.black87)),
            ],
          ),
        ),
      ),
    );
  }
}

// ── No-location state ─────────────────────────────────────────────────────────

class _NoLocation extends StatelessWidget {
  final ColorScheme cs;
  const _NoLocation({required this.cs});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.location_off_rounded, size: 48, color: cs.onSurfaceVariant),
          const SizedBox(height: 8),
          Text(
            'Location unavailable',
            style: TextStyle(fontSize: 13, color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: 4),
          Text(
            'The driver may not have started the trip yet.',
            style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
