<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Camera\StoreCameraRequest;
use App\Http\Requests\Camera\UpdateCameraRequest;
use App\Models\Camera;
use Illuminate\Http\Request;

class CameraController extends Controller
{
    public function index(Request $request)
    {
        $query = Camera::query();

        if ($request->filled('isactive')) {
            $query->where('isactive', filter_var($request->isactive, FILTER_VALIDATE_BOOLEAN));
        }

        if ($request->filled('search')) {
            $query->where('location', 'ilike', "%{$request->search}%");
        }

        return response()->json($query->orderBy('camera_id')->paginate($request->input('per_page', 20)));
    }

    public function store(StoreCameraRequest $request)
    {
        $data = $request->validated();

        $camera = Camera::create([
            'location'   => $data['location'],
            'isactive'   => $data['isactive'] ?? true,
            'code'       => $data['code'] ?? null,
            'stream_url' => $data['stream_url'] ?? null,
        ]);

        return response()->json($camera, 201);
    }

    public function show(int $id)
    {
        return response()->json(Camera::where('camera_id', $id)->withCount('events')->firstOrFail());
    }

    public function update(UpdateCameraRequest $request, int $id)
    {
        $camera = Camera::where('camera_id', $id)->firstOrFail();
        $camera->update($request->validated());

        return response()->json($camera);
    }

    public function destroy(int $id)
    {
        Camera::where('camera_id', $id)->firstOrFail()->delete();

        return response()->json(['message' => 'Camera deleted successfully.']);
    }
}
