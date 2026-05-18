<?php

namespace App\Repositories;

use App\Models\SurveillanceEvent;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;
use Illuminate\Database\Eloquent\Collection;

class SurveillanceEventRepository
{
    public function filter(array $filters, int $perPage = 20): LengthAwarePaginator
    {
        $query = SurveillanceEvent::with(['camera', 'student.user', 'section.schoolClass']);

        if (!empty($filters['camera_id'])) {
            $query->where('camera_id', $filters['camera_id']);
        }

        if (!empty($filters['detectedtype'])) {
            $query->where('detectedtype', $filters['detectedtype']);
        }

        if (!empty($filters['severity'])) {
            $query->where('severity', $filters['severity']);
        }

        if (!empty($filters['student_id'])) {
            $query->where('relatedstudent_id', $filters['student_id']);
        }

        if (!empty($filters['section_id'])) {
            $query->where('relatedsection_id', $filters['section_id']);
        }

        if (!empty($filters['status'])) {
            $query->where('status', $filters['status']);
        }

        if (!empty($filters['from'])) {
            $query->where('detectedat', '>=', $filters['from']);
        }

        if (!empty($filters['to'])) {
            $query->where('detectedat', '<=', $filters['to']);
        }

        return $query->orderByDesc('detectedat')->paginate($perPage);
    }

    public function summaryBetween(string $from, string $to, ?int $cameraId = null): Collection
    {
        $query = SurveillanceEvent::whereBetween('detectedat', [$from, $to]);

        if ($cameraId) {
            $query->where('camera_id', $cameraId);
        }

        return $query->get();
    }
}
