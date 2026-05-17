<?php

namespace App\Http\Controllers\Driver;

use App\Http\Controllers\Controller;
use App\Http\Requests\StopEvent\StoreStopEventRequest;
use App\Models\RouteStop;
use App\Models\StudentBusAssignment;
use App\Models\Trip;
use App\Models\TripStopEvent;

class StopEventController extends Controller
{
    public function index(int $driverId, int $tripId)
    {
        $trip = Trip::where('trip_id', $tripId)->where('driver_id', $driverId)->firstOrFail();

        $events = TripStopEvent::with(['stop', 'student.user'])
            ->where('trip_id', $trip->trip_id)
            ->orderBy('eventat')
            ->get();

        return response()->json($events);
    }

    public function store(StoreStopEventRequest $request, int $driverId, int $tripId)
    {
        $trip = Trip::where('trip_id', $tripId)->where('driver_id', $driverId)->firstOrFail();

        $data = $request->validated();

        $stopValid = RouteStop::where('stop_id', $data['stop_id'])
            ->where('route_id', $trip->route_id)
            ->exists();

        if (! $stopValid) {
            return response()->json(['message' => "This stop does not belong to the trip's route."], 422);
        }

        $studentValid = StudentBusAssignment::where('student_id', $data['student_id'])
            ->where('bus_id', $trip->bus_id)
            ->where('route_id', $trip->route_id)
            ->exists();

        if (! $studentValid) {
            return response()->json(['message' => 'This student is not assigned to this bus/route.'], 422);
        }

        $event = TripStopEvent::create([
            'trip_id'    => $trip->trip_id,
            'stop_id'    => $data['stop_id'],
            'student_id' => $data['student_id'],
            'eventtype'  => $data['eventtype'],
            'eventat'    => $data['eventat'] ?? now(),
        ]);

        return response()->json($event->load(['stop', 'student.user']), 201);
    }
}
