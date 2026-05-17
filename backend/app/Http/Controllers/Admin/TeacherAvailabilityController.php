<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\TeacherAvailability\StoreTeacherAvailabilityRequest;
use App\Http\Requests\TeacherAvailability\UpdateTeacherAvailabilityRequest;
use App\Models\TeacherAvailability;
use Illuminate\Http\Request;

class TeacherAvailabilityController extends Controller
{
    public function index(Request $request)
    {
        $query = TeacherAvailability::with('teacher.user');

        if ($request->filled('teacher_id'))       $query->where('teacher_id', $request->teacher_id);
        if ($request->filled('dayofweek'))         $query->where('dayofweek', $request->dayofweek);
        if ($request->filled('availabilitytype'))  $query->where('availabilitytype', $request->availabilitytype);

        $slots = $query->orderBy('teacher_id')
            ->orderByRaw("CASE dayofweek
                WHEN 'Sunday' THEN 1 WHEN 'Monday' THEN 2 WHEN 'Tuesday' THEN 3
                WHEN 'Wednesday' THEN 4 WHEN 'Thursday' THEN 5 WHEN 'Friday' THEN 6
                WHEN 'Saturday' THEN 7 END")
            ->orderBy('start_time')
            ->paginate($request->input('per_page', 50));

        return response()->json($slots);
    }

    public function store(StoreTeacherAvailabilityRequest $request)
    {
        $slot = TeacherAvailability::create($request->validated());

        return response()->json($slot->load('teacher.user'), 201);
    }

    public function show(int $id)
    {
        return response()->json(
            TeacherAvailability::where('availability_id', $id)->with('teacher.user')->firstOrFail()
        );
    }

    public function update(UpdateTeacherAvailabilityRequest $request, int $id)
    {
        $slot = TeacherAvailability::where('availability_id', $id)->firstOrFail();
        $slot->update($request->validated());

        return response()->json($slot->load('teacher.user'));
    }

    public function destroy(int $id)
    {
        TeacherAvailability::where('availability_id', $id)->firstOrFail()->delete();

        return response()->json(['message' => 'Availability slot removed successfully.']);
    }
}
