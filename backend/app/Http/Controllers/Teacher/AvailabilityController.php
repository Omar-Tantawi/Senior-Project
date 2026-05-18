<?php

namespace App\Http\Controllers\Teacher;

use App\Http\Controllers\Controller;
use App\Http\Requests\TeacherAvailability\StoreTeacherAvailabilityRequest;
use App\Http\Requests\TeacherAvailability\UpdateTeacherAvailabilityRequest;
use App\Models\TeacherAvailability;
use Illuminate\Http\Request;

class AvailabilityController extends Controller
{
    public function index(int $teacherId, Request $request)
    {
        $query = TeacherAvailability::where('teacher_id', $teacherId);

        if ($request->filled('dayofweek')) {
            $query->where('dayofweek', $request->dayofweek);
        }

        $slots = $query->orderByRaw("CASE dayofweek
                WHEN 'Sunday' THEN 1 WHEN 'Monday' THEN 2 WHEN 'Tuesday' THEN 3
                WHEN 'Wednesday' THEN 4 WHEN 'Thursday' THEN 5 WHEN 'Friday' THEN 6
                WHEN 'Saturday' THEN 7 END")
            ->orderBy('start_time')
            ->get();

        return response()->json(['teacher_id' => $teacherId, 'count' => $slots->count(), 'slots' => $slots]);
    }

    public function store(int $teacherId, StoreTeacherAvailabilityRequest $request)
    {
        $slot = TeacherAvailability::create(array_merge(
            $request->validated(),
            ['teacher_id' => $teacherId]
        ));

        return response()->json($slot, 201);
    }

    public function update(int $teacherId, int $id, UpdateTeacherAvailabilityRequest $request)
    {
        $slot = TeacherAvailability::where('teacher_id', $teacherId)
            ->where('availability_id', $id)
            ->firstOrFail();

        $slot->update($request->validated());

        return response()->json($slot);
    }

    public function destroy(int $teacherId, int $id)
    {
        TeacherAvailability::where('teacher_id', $teacherId)
            ->where('availability_id', $id)
            ->firstOrFail()
            ->delete();

        return response()->json(['message' => 'Availability slot removed successfully.']);
    }
}
