<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Bus\StoreBusRequest;
use App\Http\Requests\Bus\UpdateBusRequest;
use App\Models\Bus;
use Illuminate\Http\Request;

class BusController extends Controller
{
    public function index(Request $request)
    {
        $query = Bus::query();

        if ($search = $request->search) {
            $query->where('plate_number', 'like', "%{$search}%");
        }

        return response()->json($query->orderBy('bus_id')->paginate($request->input('per_page', 15)));
    }

    public function store(StoreBusRequest $request)
    {
        return response()->json(Bus::create($request->validated()), 201);
    }

    public function show(int $id)
    {
        return response()->json(
            Bus::with(['driverAssignments.driver.user', 'studentAssignments.student.user'])->findOrFail($id)
        );
    }

    public function update(UpdateBusRequest $request, int $id)
    {
        $bus = Bus::findOrFail($id);
        $bus->update($request->validated());

        return response()->json($bus);
    }

    public function destroy(int $id)
    {
        Bus::findOrFail($id)->delete();

        return response()->json(['message' => 'Bus deleted successfully.']);
    }
}
