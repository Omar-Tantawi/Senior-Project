<?php

namespace App\Http\Controllers\ParentControllers;

use App\Http\Controllers\Controller;
use App\Models\Enrollment;
use App\Models\Guardian;
use App\Models\ScheduleSlot;

class TeachersController extends Controller
{
    /**
     * GET /parent/{parentId}/teachers
     *
     * All distinct teachers across every child (used for general listing).
     */
    public function index(int $parentId)
    {
        $guardian = Guardian::where('parent_id', $parentId)->firstOrFail();

        $studentIds = $guardian->studentLinks()->pluck('student_id');

        $sectionIds = Enrollment::whereIn('student_id', $studentIds)
            ->where('status', 'active')
            ->pluck('section_id')
            ->unique();

        $teachers = $this->teachersForSections($sectionIds);

        return response()->json(['teachers' => $teachers]);
    }

    /**
     * GET /parent/{parentId}/children/{childId}/teachers
     *
     * Teachers who teach a specific child, each carrying the list of
     * subjects they teach that child. Used to populate the compose form.
     */
    public function forChild(int $parentId, int $childId)
    {
        // Verify the child belongs to this parent.
        $guardian = Guardian::where('parent_id', $parentId)->firstOrFail();
        $guardian->studentLinks()->where('student_id', $childId)->firstOrFail();

        $sectionId = Enrollment::where('student_id', $childId)
            ->where('status', 'active')
            ->value('section_id');

        if (! $sectionId) {
            return response()->json(['teachers' => []]);
        }

        $teachers = $this->teachersForSections([$sectionId], withSubjects: true);

        return response()->json(['teachers' => $teachers]);
    }

    // ── Shared helper ──────────────────────────────────────────────────────────

    private function teachersForSections(iterable $sectionIds, bool $withSubjects = false): \Illuminate\Support\Collection
    {
        $slots = ScheduleSlot::whereHas(
                'schedule',
                fn ($q) => $q->whereIn('section_id', $sectionIds)
            )
            ->with(['teacher.user', 'subject'])
            ->get();

        return $slots
            ->groupBy('teacher_id')
            ->map(function ($teacherSlots) use ($withSubjects) {
                $teacher = $teacherSlots->first()->teacher;
                if (! $teacher) return null;

                $entry = [
                    'id'   => $teacher->id,
                    'name' => $teacher->user->name ?? 'Unknown',
                ];

                if ($withSubjects) {
                    $entry['subjects'] = $teacherSlots
                        ->map(fn ($s) => $s->subject?->name)
                        ->filter()
                        ->unique()
                        ->values()
                        ->toArray();
                }

                return $entry;
            })
            ->filter()
            ->values();
    }
}
