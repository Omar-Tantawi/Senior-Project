<?php

namespace App\Services\Transport;

use App\Models\StudentBusAssignment;
use App\Models\Trip;

/**
 * Wall-clock simulation, reserved for future Stage 4 work.
 *
 * Stub: delegates to RealPingProvider for now so this mode degrades
 * gracefully if anyone flips TRANSPORT_MODE=simulation today.
 */
class ScheduledSimulationProvider implements PositionProvider
{
    public function __construct(private readonly RealPingProvider $fallback) {}

    public function positionFor(StudentBusAssignment $assignment, ?Trip $trip): PositionSnapshot
    {
        return $this->fallback->positionFor($assignment, $trip);
    }

    public function requiresRealTrip(): bool
    {
        return true;
    }
}
