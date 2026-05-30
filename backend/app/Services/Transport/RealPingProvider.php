<?php

namespace App\Services\Transport;

use App\Models\StudentBusAssignment;
use App\Models\TrackingPing;
use App\Models\Trip;
use App\Models\TripStopEvent;

/**
 * Real mode — reads only what the driver app posted. No synthesis.
 *
 * This is the behaviour the system had before the demo redesign.  Keeping
 * it isolated lets us switch to real GPS in one env-flag change.
 */
class RealPingProvider implements PositionProvider
{
    public function positionFor(StudentBusAssignment $assignment, ?Trip $trip): PositionSnapshot
    {
        if (! $trip) {
            return new PositionSnapshot(null, 'no_trip');
        }

        $latestTwo = TrackingPing::where('trip_id', $trip->trip_id)
            ->orderByDesc('capturedat')
            ->limit(2)
            ->get();

        $ping     = $latestTwo->first();
        $prevPing = $latestTwo->skip(1)->first();

        $lastEvent = TripStopEvent::where('trip_id', $trip->trip_id)
            ->where('student_id', $assignment->student_id)
            ->orderByDesc('eventat')
            ->first();

        $status = match (true) {
            $lastEvent?->eventtype === 'dropped' => 'dropped_off',
            $lastEvent?->eventtype === 'boarded' => 'on_bus',
            default                              => 'waiting',
        };

        $stops      = $assignment->route?->stops ?? collect();
        $homeStop   = $stops->firstWhere('stop_id', $assignment->stop_id);
        $schoolStop = $stops->last();

        $activity = ActivityResolver::resolve(
            current:  $ping     ? [(float) $ping->latitude,     (float) $ping->longitude]     : null,
            previous: $prevPing ? [(float) $prevPing->latitude, (float) $prevPing->longitude] : null,
            home:     $homeStop   ? [(float) $homeStop->latitude,   (float) $homeStop->longitude]   : null,
            school:   $schoolStop ? [(float) $schoolStop->latitude, (float) $schoolStop->longitude] : null,
        );

        return new PositionSnapshot(
            location: $ping ? [
                'latitude'   => (float) $ping->latitude,
                'longitude'  => (float) $ping->longitude,
                'capturedat' => $ping->capturedat?->toIso8601String() ?? (string) $ping->capturedat,
            ] : null,
            status: $status,
            lastEvent: $lastEvent,
            activity: $activity,
        );
    }

    public function requiresRealTrip(): bool
    {
        return true;
    }
}
