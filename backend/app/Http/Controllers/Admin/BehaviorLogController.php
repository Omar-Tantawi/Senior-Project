<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\BehaviorLog;
use App\Repositories\BehaviorLogRepository;
use Illuminate\Http\Request;

class BehaviorLogController extends Controller
{
    public function __construct(private BehaviorLogRepository $repo) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filter(
                $request->only(['teacher_id', 'student_id', 'section_id', 'type', 'from', 'to']),
                $request->input('per_page', 20)
            )
        );
    }

    public function show(int $id)
    {
        return response()->json(
            BehaviorLog::where('log_id', $id)->with(['student', 'teacher', 'section'])->firstOrFail()
        );
    }

    public function destroy(int $id)
    {
        BehaviorLog::where('log_id', $id)->firstOrFail()->delete();
        return response()->json(['message' => 'Behavior log deleted successfully.']);
    }
}
