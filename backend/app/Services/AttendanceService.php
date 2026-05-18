<?php

namespace App\Services;

use App\Models\AttendanceSession;
use App\Models\StudentAttendance;
use Illuminate\Support\Facades\DB;

class AttendanceService
{
    public function record(int $sectionId, string $date, array $records, int $capturedByUserId): array
    {
        return DB::transaction(function () use ($sectionId, $date, $records, $capturedByUserId) {
            $session = AttendanceSession::firstOrCreate([
                'section_id' => $sectionId,
                'date'       => $date,
            ]);

            $inserted = 0;
            $updated  = 0;

            foreach ($records as $record) {
                $existing = StudentAttendance::where('session_id', $session->session_id)
                    ->where('student_id', $record['student_id'])
                    ->first();

                if ($existing) {
                    $existing->update(['status' => $record['status'], 'capturedbyuserid' => $capturedByUserId]);
                    $updated++;
                } else {
                    StudentAttendance::create([
                        'session_id'       => $session->session_id,
                        'student_id'       => $record['student_id'],
                        'status'           => $record['status'],
                        'capturedbyuserid' => $capturedByUserId,
                    ]);
                    $inserted++;
                }
            }

            return [
                'message'    => 'Attendance recorded successfully.',
                'session_id' => $session->session_id,
                'date'       => $session->date,
                'inserted'   => $inserted,
                'updated'    => $updated,
            ];
        });
    }
}
