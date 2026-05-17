<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Driver\StoreDriverRequest;
use App\Http\Requests\Driver\UpdateDriverRequest;
use App\Models\Driver;
use App\Repositories\DriverRepository;

class DriverController extends Controller
{
    public function __construct(private DriverRepository $repo) {}

    public function index(\Illuminate\Http\Request $request)
    {
        return response()->json(
            $this->repo->filter($request->only(['search']), $request->input('per_page', 15))
        );
    }

    public function show(int $id)
    {
        return response()->json($this->repo->findWithProfile($id));
    }

    public function store(StoreDriverRequest $request)
    {
        $driver = $this->repo->createWithUser($request->validated());
        return response()->json($driver->load('user'), 201);
    }

    public function update(UpdateDriverRequest $request, int $id)
    {
        $driver = Driver::with('user')->findOrFail($id);
        $this->repo->updateWithUser($driver, $request->validated());
        return response()->json($driver->load('user'));
    }

    public function destroy(int $id)
    {
        $this->repo->delete(Driver::with('user')->findOrFail($id));
        return response()->json(['message' => 'Driver deleted successfully.']);
    }
}
