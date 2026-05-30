<?php

namespace App\Http\Controllers\Student;

use App\Http\Controllers\Controller;
use App\Models\Schedule;
use App\Models\Student;

class ScheduleController extends Controller
{
    /**
     * GET /student/{studentId}/schedule
     *
     * Returns the weekly timetable for the student's active enrollment section.
     */
    public function index(int $studentId)
    {
        $student = Student::with('activeEnrollment')->findOrFail($studentId);

        if (! $student->activeEnrollment) {
            return response()->json(['message' => 'No active enrollment found.'], 404);
        }

        $sectionId = $student->activeEnrollment->section_id;

        $schedule = Schedule::where('section_id', $sectionId)
            ->with(['section.schoolClass', 'slots.subject', 'slots.teacher.user'])
            ->first();

        if (! $schedule) {
            return response()->json(['message' => 'No schedule found for this section.'], 404);
        }

        // Flatten to a list of slots in the shape the Flutter app expects:
        // { slot_id, day, start_time, end_time, subject, teacher_name, order }.
        // The `scheduleslot` table has no end_time column, so we estimate it
        // as start_time + 45 minutes.
        $slots = $schedule->slots
            ->sortBy(['dayofweek', 'starttime'])
            ->values()
            ->map(function ($slot, $i) {
                $start = $slot->starttime ?? '';
                $end = $start;
                foreach (['H:i:s', 'H:i'] as $fmt) {
                    try {
                        $parsed = \Carbon\Carbon::createFromFormat($fmt, $start);
                        if ($parsed) {
                            $end = $parsed->copy()->addMinutes(45)->format($fmt);
                            break;
                        }
                    } catch (\Throwable $e) {
                        // try next format
                    }
                }
                return [
                    'slot_id'      => $slot->slot_id,
                    'day'          => strtolower($slot->dayofweek ?? ''),
                    'start_time'   => $start,
                    'end_time'     => $end,
                    'subject'      => $slot->subject->name ?? '',
                    'teacher_name' => $slot->teacher->user->name ?? '',
                    'order'        => $i,
                ];
            });

        return response()->json([
            'section' => $schedule->section->name,
            'class'   => $schedule->section->schoolClass->name,
            'term'    => $schedule->termname,
            'data'    => $slots,
        ]);
    }
}
