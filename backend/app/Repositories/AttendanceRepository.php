<?php

namespace App\Repositories;

use App\Models\AttendanceSession;
use App\Models\StudentAttendance;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class AttendanceRepository
{
    public function filterSessions(array $filters, int $perPage = 20): LengthAwarePaginator
    {
        $query = AttendanceSession::with('section.schoolClass');

        if (!empty($filters['section_id'])) {
            $query->where('section_id', $filters['section_id']);
        }

        if (!empty($filters['from'])) {
            $query->where('date', '>=', $filters['from']);
        }

        if (!empty($filters['to'])) {
            $query->where('date', '<=', $filters['to']);
        }

        return $query->orderByDesc('date')->paginate($perPage);
    }

    public function studentSummary(int $studentId, array $filters): array
    {
        $sessionsQuery = AttendanceSession::query();

        if (!empty($filters['section_id'])) {
            $sessionsQuery->where('section_id', $filters['section_id']);
        }

        if (!empty($filters['from'])) {
            $sessionsQuery->where('date', '>=', $filters['from']);
        }

        if (!empty($filters['to'])) {
            $sessionsQuery->where('date', '<=', $filters['to']);
        }

        $sessionIds = $sessionsQuery->pluck('session_id');

        $records = StudentAttendance::where('student_id', $studentId)
            ->whereIn('session_id', $sessionIds)
            ->with('session')
            ->get();

        $total   = $sessionIds->count();
        $present = $records->where('status', 'present')->count();
        $absent  = $records->where('status', 'absent')->count();
        $late    = $records->where('status', 'late')->count();
        $excused = $records->where('status', 'excused')->count();

        return [
            'student_id'     => $studentId,
            'total_sessions' => $total,
            'present'        => $present,
            'absent'         => $absent,
            'late'           => $late,
            'excused'        => $excused,
            'percentage'     => $total > 0 ? round((($present + $late) / $total) * 100, 2) : 0,
        ];
    }
}
