<?php

namespace App\Http\Controllers\ParentControllers;

use App\Http\Controllers\Controller;
use App\Models\StudentBusAssignment;
use App\Models\StudentGuardian;
use App\Models\Trip;
use App\Models\TripStopEvent;
use App\Services\Transport\PositionProvider;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class ChildBusController extends Controller
{
    /**
     * Verify this parent is linked to this student; abort otherwise.
     * Returns null on success, or a JsonResponse to abort with.
     */
    private function authorizeChild(int $parentId, int $studentId): ?JsonResponse
    {
        $linked = StudentGuardian::where('parent_id', $parentId)
            ->where('student_id', $studentId)
            ->exists();

        if (! $linked) {
            return response()->json([
                'message' => 'This student is not linked to you.',
            ], 403);
        }

        return null;
    }

    /**
     * The child's bus assignment (bus, route, pickup stop).
     */
    public function assignment(int $parentId, int $studentId)
    {
        if ($err = $this->authorizeChild($parentId, $studentId)) {
            return $err;
        }

        $assignment = StudentBusAssignment::with(['bus', 'route.stops', 'stop'])
            ->where('student_id', $studentId)
            ->first();

        if (! $assignment) {
            return response()->json(['message' => 'No bus assignment found.'], 404);
        }

        return response()->json($assignment);
    }

    /**
     * Live location of the child's bus — today's active trip, latest GPS ping.
     */
    public function liveLocation(int $parentId, int $studentId, PositionProvider $provider)
    {
        if ($err = $this->authorizeChild($parentId, $studentId)) {
            return $err;
        }

        $assignment = StudentBusAssignment::with(['bus', 'route.stops', 'stop'])
            ->where('student_id', $studentId)
            ->first();
        if (! $assignment) {
            return response()->json(['message' => 'No bus assignment found.'], 404);
        }

        // In real/simulation mode we need today's actual Trip row. In demo
        // mode any trip for this bus/route is fine — the position is fully
        // synthesised — so we relax the date filter when none exists today.
        $tripQuery = Trip::with(['driver.user', 'route'])
            ->where('bus_id', $assignment->bus_id)
            ->where('route_id', $assignment->route_id);

        $trip = (clone $tripQuery)
            ->whereDate('date', now()->toDateString())
            ->orderByDesc('trip_id')
            ->first();

        if (! $trip && ! $provider->requiresRealTrip()) {
            $trip = $tripQuery->orderByDesc('trip_id')->first();
        }

        $snapshot = $provider->positionFor($assignment, $trip);

        return response()->json([
            'trip'       => $trip,
            'location'   => $snapshot->location,
            'bus'        => $assignment->bus,
            'route'      => $assignment->route,
            'stop'       => $assignment->stop,
            'status'     => $snapshot->status,
            'activity'   => $snapshot->activity,
            'last_event' => $snapshot->lastEvent,
            'mode'       => config('transport.mode', 'demo'),
        ]);
    }

    /**
     * Child's boarding/dropoff history. Optional ?from & ?to date filters.
     */
    public function events(Request $request, int $parentId, int $studentId)
    {
        if ($err = $this->authorizeChild($parentId, $studentId)) {
            return $err;
        }

        $query = TripStopEvent::with(['stop', 'trip.route'])
            ->where('student_id', $studentId);

        if ($request->filled('from')) {
            $query->whereDate('eventat', '>=', $request->from);
        }
        if ($request->filled('to')) {
            $query->whereDate('eventat', '<=', $request->to);
        }

        return response()->json(
            $query->orderByDesc('eventat')->paginate($request->input('per_page', 20))
        );
    }
}
