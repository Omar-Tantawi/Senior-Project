<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Parent\AddChildRequest;
use App\Http\Requests\Parent\StoreParentRequest;
use App\Http\Requests\Parent\UpdateParentRequest;
use App\Models\Guardian;
use App\Models\StudentGuardian;
use App\Repositories\ParentRepository;
use Illuminate\Http\Request;

class ParentController extends Controller
{
    public function __construct(private ParentRepository $repo) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filter($request->only(['search']), $request->input('per_page', 20))
        );
    }

    public function store(StoreParentRequest $request)
    {
        $parent = $this->repo->createWithUser($request->validated());
        return response()->json($parent->load(['user', 'studentLinks.student.user']), 201);
    }

    public function show(int $id)
    {
        return response()->json($this->repo->findWithProfile($id));
    }

    public function update(UpdateParentRequest $request, int $id)
    {
        $parent = Guardian::where('parent_id', $id)->firstOrFail();
        $this->repo->updateWithUser($parent, $request->validated());
        return response()->json($parent->load(['user', 'studentLinks.student.user']));
    }

    public function destroy(int $id)
    {
        $this->repo->delete(Guardian::where('parent_id', $id)->firstOrFail());
        return response()->json(['message' => 'Parent account deleted successfully.']);
    }

    public function addChild(AddChildRequest $request, int $id)
    {
        Guardian::where('parent_id', $id)->firstOrFail();

        $data = $request->validated();

        $exists = StudentGuardian::where('parent_id', $id)->where('student_id', $data['student_id'])->exists();
        if ($exists) {
            return response()->json(['message' => 'This child is already linked to this parent.'], 422);
        }

        $uniqueRels = ['father', 'mother'];
        $newRel     = strtolower($data['relationship']);

        if (in_array($newRel, $uniqueRels, true)) {
            if (StudentGuardian::where('student_id', $data['student_id'])->whereRaw('LOWER(relationship) = ?', [$newRel])->exists()) {
                return response()->json(['message' => "This student already has a {$data['relationship']} linked. Unlink the existing one first."], 422);
            }

            $oppositeRel = $newRel === 'father' ? 'mother' : 'father';
            if (StudentGuardian::where('parent_id', $id)->whereRaw('LOWER(relationship) = ?', [$oppositeRel])->exists()) {
                return response()->json(['message' => "This parent is already registered as a {$oppositeRel} to another student and cannot also be a {$newRel}."], 422);
            }
        }

        if ($request->boolean('isprimary', false)) {
            StudentGuardian::where('student_id', $data['student_id'])->where('isprimary', true)->update(['isprimary' => false]);
        }

        $link = StudentGuardian::create([
            'student_id'   => $data['student_id'],
            'parent_id'    => $id,
            'relationship' => $data['relationship'],
            'isprimary'    => $request->boolean('isprimary', false),
        ]);

        return response()->json($link->load('student.user'), 201);
    }

    public function removeChild(int $id, int $studentId)
    {
        StudentGuardian::where('parent_id', $id)->where('student_id', $studentId)->firstOrFail()->delete();
        return response()->json(['message' => 'Child unlinked successfully.']);
    }
}
