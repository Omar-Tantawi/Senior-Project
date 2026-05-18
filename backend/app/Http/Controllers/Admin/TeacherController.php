<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Teacher\StoreTeacherRequest;
use App\Http\Requests\Teacher\UpdateTeacherRequest;
use App\Models\Teacher;
use App\Repositories\TeacherRepository;

class TeacherController extends Controller
{
    public function __construct(private TeacherRepository $repo) {}

    public function index(\Illuminate\Http\Request $request)
    {
        return response()->json(
            $this->repo->filter($request->only(['search', 'status', 'subject_id']), $request->input('per_page', 15))
        );
    }

    public function show(int $id)
    {
        return response()->json($this->repo->findWithProfile($id));
    }

    public function store(StoreTeacherRequest $request)
    {
        $teacher = $this->repo->createWithUser($request->validated());
        return response()->json($teacher->load('user'), 201);
    }

    public function update(UpdateTeacherRequest $request, int $id)
    {
        $teacher = Teacher::with('user')->findOrFail($id);
        $this->repo->updateWithUser($teacher, $request->validated());
        return response()->json($teacher->load('user'));
    }

    public function destroy(int $id)
    {
        $this->repo->delete(Teacher::with('user')->findOrFail($id));
        return response()->json(['message' => 'Teacher deleted successfully.']);
    }
}
