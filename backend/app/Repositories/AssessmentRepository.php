<?php

namespace App\Repositories;

use App\Models\Assessment;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class AssessmentRepository
{
    public function filter(array $filters, int $perPage = 15): LengthAwarePaginator
    {
        $query = Assessment::with(['subject', 'section.schoolClass']);

        if (!empty($filters['subject_id'])) {
            $query->where('subject_id', $filters['subject_id']);
        }

        if (!empty($filters['section_id'])) {
            $query->where('section_id', $filters['section_id']);
        }

        if (!empty($filters['assessmenttype'])) {
            $query->where('assessmenttype', $filters['assessmenttype']);
        }

        return $query->latest('date')->paginate($perPage);
    }
}
