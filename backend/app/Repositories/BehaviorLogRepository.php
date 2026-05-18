<?php

namespace App\Repositories;

use App\Models\BehaviorLog;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class BehaviorLogRepository
{
    public function filter(array $filters, int $perPage = 20): LengthAwarePaginator
    {
        $query = BehaviorLog::with(['student', 'teacher', 'section']);

        if (!empty($filters['teacher_id'])) {
            $query->where('teacher_id', $filters['teacher_id']);
        }

        if (!empty($filters['student_id'])) {
            $query->where('student_id', $filters['student_id']);
        }

        if (!empty($filters['section_id'])) {
            $query->where('section_id', $filters['section_id']);
        }

        if (!empty($filters['type'])) {
            $query->where('type', $filters['type']);
        }

        if (!empty($filters['from'])) {
            $query->where('date', '>=', $filters['from']);
        }

        if (!empty($filters['to'])) {
            $query->where('date', '<=', $filters['to']);
        }

        return $query->orderByDesc('date')->paginate($perPage);
    }
}
