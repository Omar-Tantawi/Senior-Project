<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Attendance\StoreAttendanceRequest;
use App\Http\Requests\Attendance\UpdateAttendanceRecordRequest;
use App\Models\AttendanceSession;
use App\Models\StudentAttendance;
use App\Repositories\AttendanceRepository;
use App\Services\AttendanceService;
use Illuminate\Http\Request;

class AttendanceController extends Controller
{
    public function __construct(
        private AttendanceRepository $repo,
        private AttendanceService $service
    ) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filterSessions(
                $request->only(['section_id', 'from', 'to']),
                $request->input('per_page', 20)
            )
        );
    }

    public function show(int $sessionId)
    {
        $session = AttendanceSession::where('session_id', $sessionId)
            ->with('section.schoolClass')
            ->firstOrFail();

        $records = StudentAttendance::where('session_id', $sessionId)->with('student.user')->get();

        return response()->json([
            'session'  => $session,
            'summary'  => [
                'present' => $records->where('status', 'present')->count(),
                'absent'  => $records->where('status', 'absent')->count(),
                'late'    => $records->where('status', 'late')->count(),
                'excused' => $records->where('status', 'excused')->count(),
            ],
            'students' => $records->map(fn($r) => [
                'attendance_id' => $r->attendance_id,
                'student_id'    => $r->student_id,
                'name'          => $r->student->user->name,
                'status'        => $r->status,
            ]),
        ]);
    }

    public function store(StoreAttendanceRequest $request)
    {
        $data   = $request->validated();
        $result = $this->service->record($data['section_id'], $data['date'], $data['records'], $request->user()->id);
        return response()->json($result, 201);
    }

    public function updateRecord(UpdateAttendanceRecordRequest $request, int $sessionId, int $attendanceId)
    {
        $record = StudentAttendance::where('session_id', $sessionId)
            ->where('attendance_id', $attendanceId)
            ->firstOrFail();

        $record->update([
            'status'           => $request->validated()['status'],
            'capturedbyuserid' => $request->user()->id,
        ]);

        return response()->json([
            'message'       => 'Attendance record updated.',
            'attendance_id' => $record->attendance_id,
            'status'        => $record->status,
        ]);
    }

    public function destroy(int $sessionId)
    {
        $session = AttendanceSession::where('session_id', $sessionId)->firstOrFail();
        StudentAttendance::where('session_id', $sessionId)->delete();
        $session->delete();
        return response()->json(['message' => 'Attendance session deleted successfully.']);
    }

    public function studentSummary(int $studentId, Request $request)
    {
        return response()->json(
            $this->repo->studentSummary($studentId, $request->only(['section_id', 'from', 'to']))
        );
    }
}
