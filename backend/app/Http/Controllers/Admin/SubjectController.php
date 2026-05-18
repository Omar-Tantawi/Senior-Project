<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Subject\StoreSubjectRequest;
use App\Http\Requests\Subject\UpdateSubjectRequest;
use App\Models\Subject;

class SubjectController extends Controller
{
    public function index()
    {
        return response()->json(Subject::orderBy('name')->get());
    }

    public function store(StoreSubjectRequest $request)
    {
        return response()->json(Subject::create($request->validated()), 201);
    }

    public function show(int $id)
    {
        return response()->json(Subject::findOrFail($id));
    }

    public function update(UpdateSubjectRequest $request, int $id)
    {
        $subject = Subject::findOrFail($id);
        $subject->update($request->validated());

        return response()->json($subject);
    }

    public function destroy(int $id)
    {
        Subject::findOrFail($id)->delete();

        return response()->json(['message' => 'Subject deleted successfully.']);
    }
}
