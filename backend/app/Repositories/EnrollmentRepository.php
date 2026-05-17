<?php

namespace App\Repositories;

use App\Models\Enrollment;
use Illuminate\Database\Eloquent\Collection;

class EnrollmentRepository
{
    public function filter(array $filters): Collection
    {
        $query = Enrollment::with(['student.user', 'section.schoolClass.schoolYear']);

        if (!empty($filters['section_id'])) {
            $query->where('section_id', $filters['section_id']);
        }

        if (!empty($filters['student_id'])) {
            $query->where('student_id', $filters['student_id']);
        }

        if (!empty($filters['status'])) {
            $query->where('status', $filters['status']);
        }

        return $query->get();
    }

    public function existsInSection(int $studentId, int $sectionId): bool
    {
        return Enrollment::where('student_id', $studentId)
            ->where('section_id', $sectionId)
            ->exists();
    }
}
