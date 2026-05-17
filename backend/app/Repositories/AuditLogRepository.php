<?php

namespace App\Repositories;

use App\Models\AuditLog;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class AuditLogRepository
{
    public function filter(array $filters, int $perPage = 30): LengthAwarePaginator
    {
        $query = AuditLog::query();

        if (!empty($filters['user_id'])) {
            $query->where('user_id', $filters['user_id']);
        }

        if (!empty($filters['role'])) {
            $query->where('role', $filters['role']);
        }

        if (!empty($filters['action'])) {
            $query->where('action', strtoupper($filters['action']));
        }

        if (!empty($filters['resource'])) {
            $query->where('resource', $filters['resource']);
        }

        if (!empty($filters['resource_id'])) {
            $query->where('resource_id', $filters['resource_id']);
        }

        if (!empty($filters['from'])) {
            $query->where('performed_at', '>=', $filters['from']);
        }

        if (!empty($filters['to'])) {
            $query->where('performed_at', '<=', $filters['to']);
        }

        if (!empty($filters['search'])) {
            $query->where('user_name', 'ilike', "%{$filters['search']}%");
        }

        return $query->orderByDesc('performed_at')->paginate($perPage);
    }
}
