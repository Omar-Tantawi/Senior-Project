<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Enrollment\StoreEnrollmentRequest;
use App\Http\Requests\Enrollment\UpdateEnrollmentRequest;
use App\Models\Enrollment;
use App\Repositories\EnrollmentRepository;

class EnrollmentController extends Controller
{
    public function __construct(private EnrollmentRepository $repo) {}

    public function index(\Illuminate\Http\Request $request)
    {
        return response()->json(
            $this->repo->filter($request->only(['section_id', 'student_id', 'status']))
        );
    }

    public function store(StoreEnrollmentRequest $request)
    {
        $data = $request->validated();

        // Prevent duplicate in the same section
        if ($this->repo->existsInSection($data['student_id'], $data['section_id'])) {
            return response()->json(['message' => 'Student is already enrolled in this section.'], 422);
        }

        // Prevent enrolling in a second section while already active in one
        $active = $this->repo->activeEnrollment($data['student_id']);
        if ($active) {
            $where = $active->section->schoolClass->name . ' — ' . $active->section->name;
            return response()->json([
                'message' => "Student is already enrolled in {$where}. Remove them first before assigning a new section.",
            ], 422);
        }

        $enrollment = Enrollment::create([
            'student_id' => $data['student_id'],
            'section_id' => $data['section_id'],
            'status'     => $data['status'] ?? 'active',
        ]);

        return response()->json($enrollment->load(['student.user', 'section.schoolClass']), 201);
    }

    public function update(UpdateEnrollmentRequest $request, int $id)
    {
        $enrollment = Enrollment::findOrFail($id);
        $enrollment->update(['status' => $request->validated()['status']]);
        return response()->json($enrollment->load(['student.user', 'section.schoolClass']));
    }

    public function destroy(int $id)
    {
        Enrollment::findOrFail($id)->delete();
        return response()->json(['message' => 'Enrollment removed successfully.']);
    }
}
