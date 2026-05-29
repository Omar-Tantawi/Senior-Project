<?php

namespace App\Http\Controllers\Ai;

use App\Http\Controllers\Controller;
use App\Models\Student;
use App\Services\AttendanceService;
use Illuminate\Http\Request;

class AttendanceController extends Controller
{
    public function __construct(private AttendanceService $service) {}

    /**
     * GET /ai/attendance/students
     * Returns all active students in the format the AI expects.
     */
    public function students()
    {
        $students = Student::with('user')
            ->where('status', 'active')
            ->get()
            ->map(fn($s) => [
                'student_id'   => (string) $s->id,
                'student_name' => $s->user->name,
            ]);

        return response()->json($students);
    }

    /**
     * POST /ai/attendance
     * Receives an AI attendance report and stores it.
     *
     * Body:
     *   section_id  (required) int
     *   date        (required) date string
     *   attendance  (required) array of {student_id, status, student_name?}
     */
    public function store(Request $request)
    {
        $data = $request->validate([
            'section_id'              => 'required|integer|exists:section,section_id',
            'date'                    => 'required|date',
            'attendance'              => 'required|array|min:1',
            'attendance.*.student_id' => 'required',
            'attendance.*.status'     => 'required|in:present,absent,late,excused',
        ]);

        $records = array_map(fn($r) => [
            'student_id' => (int) $r['student_id'],
            'status'     => $r['status'],
        ], $data['attendance']);

        // capturedbyuserid = null → AI-submitted (nullable FK, SET NULL on delete)
        $result = $this->service->record(
            $data['section_id'],
            $data['date'],
            $records,
            null
        );

        return response()->json($result, 201);
    }
}
