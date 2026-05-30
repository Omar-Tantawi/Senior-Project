<?php

namespace App\Services\Transport;

use App\Models\TripStopEvent;

/**
 * The full live-tracking payload returned by any PositionProvider.
 *
 * Shape is intentionally identical across all three modes so the parent
 * controller and the Flutter client cannot tell them apart.
 */
class PositionSnapshot
{
    public function __construct(
        /** ['latitude' => float, 'longitude' => float, 'capturedat' => string] or null */
        public readonly ?array $location,
        /** waiting | on_bus | dropped_off | no_trip */
        public readonly string $status,
        public readonly ?TripStopEvent $lastEvent = null,
        public readonly bool $isSynthetic = false,
        /** at_home | heading_to_school | at_school | heading_home | unknown */
        public readonly string $activity = 'unknown',
    ) {}
}
