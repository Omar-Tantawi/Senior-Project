<?php

namespace App\Http\Controllers\Teacher;

use App\Http\Controllers\Controller;
use App\Http\Requests\VacationRequest\StoreVacationRequestRequest;
use App\Models\VacationRequest;
use Illuminate\Http\Request;

class VacationRequestController extends Controller
{
    public function index(int $teacherId, Request $request)
    {
        $query = VacationRequest::where('teacher_id', $teacherId);

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        $requests = $query->orderByDesc('start_date')->get();

        return response()->json(['teacher_id' => $teacherId, 'count' => $requests->count(), 'requests' => $requests]);
    }

    public function store(int $teacherId, StoreVacationRequestRequest $request)
    {
        $data = $request->validated();

        $overlap = VacationRequest::where('teacher_id', $teacherId)
            ->where('status', '!=', 'rejected')
            ->where(function ($q) use ($data) {
                $q->whereBetween('start_date', [$data['start_date'], $data['end_date']])
                  ->orWhereBetween('end_date', [$data['start_date'], $data['end_date']])
                  ->orWhere(function ($q2) use ($data) {
                      $q2->where('start_date', '<=', $data['start_date'])
                         ->where('end_date', '>=', $data['end_date']);
                  });
            })
            ->exists();

        if ($overlap) {
            return response()->json([
                'message' => 'You already have a vacation request that overlaps with these dates.',
            ], 422);
        }

        $vacation = VacationRequest::create([
            'teacher_id' => $teacherId,
            'start_date' => $data['start_date'],
            'end_date'   => $data['end_date'],
            'status'     => 'pending',
        ]);

        return response()->json($vacation, 201);
    }

    public function show(int $teacherId, int $id)
    {
        return response()->json(
            VacationRequest::where('teacher_id', $teacherId)->where('vacation_id', $id)->firstOrFail()
        );
    }

    public function destroy(int $teacherId, int $id)
    {
        VacationRequest::where('teacher_id', $teacherId)
            ->where('vacation_id', $id)
            ->where('status', 'pending')
            ->firstOrFail()
            ->delete();

        return response()->json(['message' => 'Vacation request cancelled successfully.']);
    }
}
