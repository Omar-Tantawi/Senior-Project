<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\AuditLog;
use App\Repositories\AuditLogRepository;
use Illuminate\Http\Request;

class AuditLogController extends Controller
{
    public function __construct(private AuditLogRepository $repo) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filter(
                $request->only(['user_id', 'role', 'action', 'resource', 'resource_id', 'from', 'to', 'search']),
                $request->input('per_page', 30)
            )
        );
    }

    public function show(int $id)
    {
        return response()->json(AuditLog::findOrFail($id));
    }

    public function userHistory(int $userId, Request $request)
    {
        $logs = AuditLog::where('user_id', $userId)
            ->orderByDesc('performed_at')
            ->paginate($request->input('per_page', 30));

        return response()->json($logs);
    }

    public function resourceHistory(string $resource, string $resourceId, Request $request)
    {
        $logs = AuditLog::where('resource', $resource)
            ->where('resource_id', $resourceId)
            ->orderByDesc('performed_at')
            ->paginate($request->input('per_page', 30));

        return response()->json($logs);
    }
}
