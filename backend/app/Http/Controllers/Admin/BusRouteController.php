<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\BusRoute\StoreBusRouteRequest;
use App\Http\Requests\BusRoute\UpdateBusRouteRequest;
use App\Models\BusRoute;
use Illuminate\Http\Request;

class BusRouteController extends Controller
{
    public function index(Request $request)
    {
        $query = BusRoute::with('stops');

        if ($search = $request->search) {
            $query->where('name', 'like', "%{$search}%");
        }

        return response()->json($query->orderBy('name')->paginate($request->input('per_page', 15)));
    }

    public function store(StoreBusRouteRequest $request)
    {
        return response()->json(BusRoute::create($request->validated()), 201);
    }

    public function show(int $id)
    {
        return response()->json(BusRoute::with('stops')->findOrFail($id));
    }

    public function update(UpdateBusRouteRequest $request, int $id)
    {
        $route = BusRoute::findOrFail($id);
        $route->update($request->validated());

        return response()->json($route);
    }

    public function destroy(int $id)
    {
        BusRoute::findOrFail($id)->delete();

        return response()->json(['message' => 'Route deleted successfully.']);
    }
}
