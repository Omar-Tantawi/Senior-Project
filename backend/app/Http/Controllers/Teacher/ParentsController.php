<?php

namespace App\Http\Controllers\Teacher;

use App\Http\Controllers\Controller;
use App\Models\Enrollment;
use App\Models\ScheduleSlot;
use App\Models\StudentGuardian;
use App\Models\Teacher;
use Illuminate\Support\Collection;

class ParentsController extends Controller
{
    /**
     * Return the parents of all students taught by this teacher,
     * grouped by parent so each parent appears once with all their children.
     *
     * GET /api/teacher/{teacherId}/parents
     * Response: { parents: [ { user_id, name, students: [{ id, name }] } ] }
     */
    public function index(int $teacherId)
    {
        $teacher = Teacher::findOrFail($teacherId);

        // 1. Sections taught by this teacher (ScheduleSlot has a direct teacher_id)
        $sectionIds = ScheduleSlot::where('teacher_id', $teacher->id)
            ->with('schedule')
            ->get()
            ->pluck('schedule.section_id')
            ->filter()
            ->unique()
            ->values();

        if ($sectionIds->isEmpty()) {
            return response()->json(['parents' => []]);
        }

        // 2. Active students enrolled in those sections
        $studentIds = Enrollment::whereIn('section_id', $sectionIds)
            ->where('status', 'active')
            ->pluck('student_id')
            ->unique();

        if ($studentIds->isEmpty()) {
            return response()->json(['parents' => []]);
        }

        // 3. Guardian links for those students, grouped by parent
        $links = StudentGuardian::whereIn('student_id', $studentIds)
            ->with(['guardian.user', 'student.user'])
            ->get();

        $parents = $links->groupBy('parent_id')->map(function (Collection $parentLinks) {
            $guardian = $parentLinks->first()->guardian;
            if (! $guardian || ! $guardian->user) {
                return null;
            }

            $students = $parentLinks->map(function ($link) {
                $student = $link->student;
                if (! $student || ! $student->user) {
                    return null;
                }
                return [
                    'id'   => $student->id,
                    'name' => $student->user->name ?? 'Unknown',
                ];
            })->filter()->values();

            return [
                'user_id'  => $guardian->user->id,
                'name'     => $guardian->user->name,
                'students' => $students,
            ];
        })->filter()->values();

        return response()->json(['parents' => $parents]);
    }
}
