<?php

namespace App\Repositories;

use App\Models\VacationRequest;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class VacationRequestRepository
{
    public function filter(array $filters, int $perPage = 20): LengthAwarePaginator
    {
        $query = VacationRequest::with('teacher.user');

        if (!empty($filters['teacher_id'])) {
            $query->where('teacher_id', $filters['teacher_id']);
        }

        if (!empty($filters['status'])) {
            $query->where('status', $filters['status']);
        }

        if (!empty($filters['from'])) {
            $query->where('start_date', '>=', $filters['from']);
        }

        if (!empty($filters['to'])) {
            $query->where('end_date', '<=', $filters['to']);
        }

        return $query->orderByDesc('start_date')->paginate($perPage);
    }
}
