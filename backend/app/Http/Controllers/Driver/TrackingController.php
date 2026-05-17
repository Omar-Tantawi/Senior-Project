<?php

namespace App\Http\Controllers\Driver;

use App\Http\Controllers\Controller;
use App\Http\Requests\Tracking\StoreTrackingPingRequest;
use App\Models\TrackingPing;
use App\Models\Trip;

class TrackingController extends Controller
{
    public function store(StoreTrackingPingRequest $request, int $driverId, int $tripId)
    {
        $trip = Trip::where('trip_id', $tripId)->where('driver_id', $driverId)->firstOrFail();

        $data = $request->validated();

        $ping = TrackingPing::create([
            'trip_id'    => $trip->trip_id,
            'latitude'   => $data['latitude'],
            'longitude'  => $data['longitude'],
            'capturedat' => $data['capturedat'] ?? now(),
        ]);

        return response()->json($ping, 201);
    }
}
