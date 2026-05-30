<?php

namespace App\Services\Transport;

use App\Models\StudentBusAssignment;
use App\Models\Trip;

/**
 * Contract for "where is the bus right now and what is this student's status?"
 *
 * Implementations:
 *   - DemoLoopProvider           — virtual loop, ignores wall clock
 *   - ScheduledSimulationProvider — wall clock + morning/afternoon windows
 *   - RealPingProvider           — only real TrackingPing rows
 */
interface PositionProvider
{
    /**
     * @param StudentBusAssignment $assignment   The student's bus/route/stop link, with route.stops eager-loaded.
     * @param Trip|null            $trip         Today's trip if one exists. May be null in real mode.
     */
    public function positionFor(StudentBusAssignment $assignment, ?Trip $trip): PositionSnapshot;

    /**
     * Whether this provider needs a real Trip row to function.
     *
     * Demo mode returns false (synthesises everything); real mode returns true.
     */
    public function requiresRealTrip(): bool;
}
