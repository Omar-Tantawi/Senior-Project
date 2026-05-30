<?php

namespace App\Services\Transport;

/**
 * Pure geometry — given a current GPS point and a recent prior one, decide
 * what the bus is doing.  No clocks involved.
 *
 *   at_home            — within ~80 m of the student's pickup stop
 *   at_school          — within ~80 m of the school
 *   heading_to_school  — moving and getting closer to the school
 *   heading_home       — moving and getting closer to home
 *   unknown            — not enough information
 *
 * This is the single source of truth across demo / simulation / real modes.
 */
class ActivityResolver
{
    /** Distance threshold (m) under which the bus counts as "at" a location. */
    public const AT_LOCATION_METERS = 80.0;

    /**
     * @param array{0: float, 1: float}|null $current   [lat, lng]
     * @param array{0: float, 1: float}|null $previous  [lat, lng]
     * @param array{0: float, 1: float}|null $home      pickup stop [lat, lng]
     * @param array{0: float, 1: float}|null $school    school stop [lat, lng]
     */
    public static function resolve(
        ?array $current,
        ?array $previous,
        ?array $home,
        ?array $school,
    ): string {
        if ($current === null || $home === null || $school === null) {
            return 'unknown';
        }

        $distToHome   = self::haversine($current, $home);
        $distToSchool = self::haversine($current, $school);

        // Within the "parked" radius wins outright — direction doesn't matter.
        if ($distToHome   < self::AT_LOCATION_METERS) return 'at_home';
        if ($distToSchool < self::AT_LOCATION_METERS) return 'at_school';

        if ($previous === null) {
            return 'unknown';
        }

        $prevToSchool = self::haversine($previous, $school);
        $prevToHome   = self::haversine($previous, $home);

        // Whichever endpoint the bus is *approaching* wins.
        $gainOnSchool = $prevToSchool - $distToSchool; // positive = closing in
        $gainOnHome   = $prevToHome   - $distToHome;

        if ($gainOnSchool > $gainOnHome) return 'heading_to_school';
        if ($gainOnHome   > $gainOnSchool) return 'heading_home';

        return 'unknown';
    }

    /**
     * Great-circle distance in metres between two [lat, lng] pairs.
     */
    public static function haversine(array $a, array $b): float
    {
        $earthM = 6_371_000.0;
        $lat1   = deg2rad((float) $a[0]);
        $lat2   = deg2rad((float) $b[0]);
        $dLat   = $lat2 - $lat1;
        $dLng   = deg2rad((float) $b[1] - (float) $a[1]);

        $h = sin($dLat / 2) ** 2 + cos($lat1) * cos($lat2) * sin($dLng / 2) ** 2;
        return 2 * $earthM * asin(min(1.0, sqrt($h)));
    }
}
