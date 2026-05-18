<?php

namespace App\Http\Controllers\ParentControllers;

use App\Http\Controllers\Controller;
use App\Http\Requests\Complaint\StoreComplaintRequest;
use App\Models\Complaint;
use App\Models\Guardian;
use Illuminate\Http\Request;

class ComplaintController extends Controller
{
    public function index(int $parentId, Request $request)
    {
        $query = Complaint::where('parent_id', $parentId)->with('student.user');

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        return response()->json($query->orderByDesc('created_at')->paginate($request->input('per_page', 20)));
    }

    public function store(int $parentId, StoreComplaintRequest $request)
    {
        $guardian = Guardian::where('parent_id', $parentId)->firstOrFail();
        $data     = $request->validated();

        if (! empty($data['student_id'])) {
            $guardian->studentLinks()->where('student_id', $data['student_id'])->firstOrFail();
        }

        $complaint = Complaint::create([
            'parent_id'  => $parentId,
            'student_id' => $data['student_id'] ?? null,
            'subject'    => $data['subject'],
            'body'       => $data['body'],
        ]);

        return response()->json($complaint->load('student.user'), 201);
    }

    public function show(int $parentId, int $id)
    {
        return response()->json(
            Complaint::where('complaint_id', $id)->where('parent_id', $parentId)->with('student.user')->firstOrFail()
        );
    }
}
