<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\RouteStop\StoreRouteStopRequest;
use App\Http\Requests\RouteStop\UpdateRouteStopRequest;
use App\Models\RouteStop;
use Illuminate\Http\Request;

class RouteStopController extends Controller
{
    public function index(Request $request)
    {
        $query = RouteStop::with('route');

        if ($request->filled('route_id')) {
            $query->where('route_id', $request->route_id);
        }

        return response()->json($query->orderBy('route_id')->orderBy('stoporder')->get());
    }

    public function store(StoreRouteStopRequest $request)
    {
        $data = $request->validated();

        $clash = RouteStop::where('route_id', $data['route_id'])
            ->where('stoporder', $data['stoporder'])
            ->exists();

        if ($clash) {
            return response()->json(['message' => 'A stop with this order already exists on this route.'], 422);
        }

        return response()->json(RouteStop::create($data), 201);
    }

    public function show(int $id)
    {
        return response()->json(RouteStop::with('route')->findOrFail($id));
    }

    public function update(UpdateRouteStopRequest $request, int $id)
    {
        $stop = RouteStop::findOrFail($id);
        $data = $request->validated();

        if (isset($data['stoporder']) && $data['stoporder'] != $stop->stoporder) {
            $clash = RouteStop::where('route_id', $stop->route_id)
                ->where('stoporder', $data['stoporder'])
                ->where('stop_id', '!=', $id)
                ->exists();

            if ($clash) {
                return response()->json(['message' => 'A stop with this order already exists on this route.'], 422);
            }
        }

        $stop->update($data);

        return response()->json($stop);
    }

    public function destroy(int $id)
    {
        RouteStop::findOrFail($id)->delete();

        return response()->json(['message' => 'Stop deleted successfully.']);
    }
}
