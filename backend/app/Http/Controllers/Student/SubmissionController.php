<?php

namespace App\Http\Controllers\Student;

use App\Http\Controllers\Controller;
use App\Http\Requests\Submission\StoreSubmissionRequest;
use App\Models\HomeworkSubmission;
use Illuminate\Http\Request;

class SubmissionController extends Controller
{
    public function index(int $studentId, Request $request)
    {
        $query = HomeworkSubmission::where('student_id', $studentId)
            ->with(['homework.subject', 'homework.section.schoolClass']);

        if ($request->filled('homework_id')) $query->where('homework_id', $request->homework_id);
        if ($request->filled('status'))      $query->where('status', $request->status);

        return response()->json($query->orderByDesc('submittedat')->paginate($request->input('per_page', 20)));
    }

    public function store(int $studentId, StoreSubmissionRequest $request)
    {
        $data = $request->validated();

        $exists = HomeworkSubmission::where('homework_id', $data['homework_id'])
            ->where('student_id', $studentId)
            ->exists();

        if ($exists) {
            return response()->json(['message' => 'You have already submitted this homework.'], 422);
        }

        $filePath = null;
        if ($request->hasFile('file')) {
            $filePath = $request->file('file')->store(
                "submissions/{$data['homework_id']}/{$studentId}",
                'public'
            );
        }

        $submission = HomeworkSubmission::create([
            'homework_id' => $data['homework_id'],
            'student_id'  => $studentId,
            'submittedat' => now(),
            'status'      => 'submitted',
            'file_path'   => $filePath,
        ]);

        return response()->json($submission->load(['homework.subject']), 201);
    }

    public function show(int $studentId, int $id)
    {
        return response()->json(
            HomeworkSubmission::where('submission_id', $id)
                ->where('student_id', $studentId)
                ->with(['homework.subject', 'homework.teacher.user'])
                ->firstOrFail()
        );
    }
}
