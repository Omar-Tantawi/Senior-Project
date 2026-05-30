<?php

namespace App\Services\Transport;

use App\Models\StudentBusAssignment;
use App\Models\Trip;

/**
 * Demo mode — the bus walks Stop 1 → Stop 2 → … → School and back, forever.
 *
 * Position is a pure function of (wall-clock seconds mod LOOP_SECONDS), so
 * every request the examiner makes shows a moving bus, no matter the date,
 * time, weekend, or holiday.
 *
 * No DB writes. No driver app required. No seeded pings consulted.
 */
class DemoLoopProvider implements PositionProvider
{
    public function positionFor(StudentBusAssignment $assignment, ?Trip $trip): PositionSnapshot
    {
        $stops = $assignment->route?->stops ?? collect();
        if ($stops->count() < 2) {
            return new PositionSnapshot(null, 'no_trip', null, true);
        }

        $loopSeconds = (int) config('transport.demo.loop_seconds', 120);
        $phaseOffset = (int) config('transport.demo.bus_phase_offset', 30);
        $busPhase    = (int) ($assignment->bus_id ?? 0) * $phaseOffset;

        $now     = time();
        [$lat, $lng, $forward, $legProgress]
            = $this->positionAt($assignment, $now + $busPhase, $loopSeconds);

        // Compute the position a short while ago so the activity resolver has
        // a direction of motion to work with (without consulting any clock).
        [$prevLat, $prevLng]
            = $this->positionAt($assignment, $now + $busPhase - 15, $loopSeconds);

        $status = $this->statusFor($assignment, $stops, $forward, $legProgress);

        $homeStop   = $stops->firstWhere('stop_id', $assignment->stop_id);
        $schoolStop = $stops->last();

        $activity = ActivityResolver::resolve(
            current:  [$lat, $lng],
            previous: [$prevLat, $prevLng],
            home:     $homeStop ? [(float) $homeStop->latitude,   (float) $homeStop->longitude]   : null,
            school:   $schoolStop ? [(float) $schoolStop->latitude, (float) $schoolStop->longitude] : null,
        );

        return new PositionSnapshot(
            location: [
                'latitude'   => round($lat, 7),
                'longitude'  => round($lng, 7),
                'capturedat' => now()->toIso8601String(),
            ],
            status: $status,
            lastEvent: null,
            isSynthetic: true,
            activity: $activity,
        );
    }

    /**
     * Pure: where is the bus at virtual time $t (any unix-seconds value)?
     *
     * @return array{0: float, 1: float, 2: bool, 3: float}
     *         [lat, lng, forward, legProgress]
     */
    private function positionAt(
        StudentBusAssignment $assignment,
        int $t,
        int $loopSeconds,
    ): array {
        $t      = (($t % $loopSeconds) + $loopSeconds) % $loopSeconds;
        $phase  = $t / $loopSeconds;
        $forward      = $phase < 0.5;
        $legProgress  = $forward ? ($phase * 2.0) : (($phase - 0.5) * 2.0);

        $polyline = $assignment->route?->polyline;

        if (is_array($polyline) && count($polyline) >= 2) {
            $path = $forward ? $polyline : array_reverse($polyline);
            [$lat, $lng] = $this->lerpPolyline($path, $legProgress);
        } else {
            $stops          = $assignment->route?->stops ?? collect();
            $sequence       = $forward ? $stops->values() : $stops->reverse()->values();
            $segments       = max(1, $sequence->count() - 1);
            $busSegment     = (int) floor($legProgress * $segments);
            $busSegment     = min($busSegment, $segments - 1);
            $segmentLocal   = ($legProgress * $segments) - $busSegment;

            $from = $sequence[$busSegment];
            $to   = $sequence[$busSegment + 1];
            $lat  = $from->latitude + ($to->latitude - $from->latitude) * $segmentLocal;
            $lng  = $from->longitude + ($to->longitude - $from->longitude) * $segmentLocal;
        }

        return [(float) $lat, (float) $lng, $forward, $legProgress];
    }

    public function requiresRealTrip(): bool
    {
        return false;
    }

    /**
     * Walk a list of [lat, lng] waypoints with linear interpolation so that
     * `progress = 0.0` is the first point and `progress = 1.0` is the last.
     *
     * @param array<int, array{0: float, 1: float}> $points
     * @return array{0: float, 1: float}
     */
    private function lerpPolyline(array $points, float $progress): array
    {
        $progress = max(0.0, min(1.0, $progress));
        $segments = count($points) - 1;
        $pos      = $progress * $segments;
        $i        = (int) floor($pos);
        if ($i >= $segments) {
            return [(float) $points[$segments][0], (float) $points[$segments][1]];
        }
        $local = $pos - $i;
        $a     = $points[$i];
        $b     = $points[$i + 1];
        return [
            (float) $a[0] + ((float) $b[0] - (float) $a[0]) * $local,
            (float) $a[1] + ((float) $b[1] - (float) $a[1]) * $local,
        ];
    }

    /**
     * Map loop position → realistic boarding status for one student.
     */
    private function statusFor(
        StudentBusAssignment $assignment,
        \Illuminate\Support\Collection $stops,
        bool $forward,
        float $legProgress,
    ): string {
        $studentStopId = $assignment->stop_id;
        $stopsArr      = $stops->values();
        $schoolIndex   = $stopsArr->count() - 1;

        // Where in the route order is this student's pickup stop?
        $pickupIndex = $stopsArr->search(fn ($s) => $s->stop_id === $studentStopId);
        if ($pickupIndex === false) {
            $pickupIndex = 0;
        }

        $segments       = $schoolIndex; // 0..N-1
        $busSegmentPos  = $legProgress * $segments;  // continuous 0..N-1

        // Wider thresholds keep each status visible long enough for a viva
        // examiner to notice the transition.
        $arrivalBand = 0.25;

        if ($forward) {
            // Forward leg: home → school
            if ($busSegmentPos < $pickupIndex - $arrivalBand) {
                return 'waiting';
            }
            if ($busSegmentPos < $segments - $arrivalBand) {
                return 'on_bus';
            }
            return 'dropped_off'; // arriving at / parked at school
        }

        // Reverse leg: school → home — convert reverse position back to the
        // original stop ordering.
        $busOriginalIndex = $schoolIndex - $busSegmentPos;

        if ($busOriginalIndex > $pickupIndex + $arrivalBand) {
            return 'on_bus';      // still riding home
        }
        return 'dropped_off';     // dropped at or past home
    }
}
