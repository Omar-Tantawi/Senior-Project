<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Student\StoreStudentRequest;
use App\Http\Requests\Student\UpdateStudentRequest;
use App\Repositories\StudentRepository;
use Illuminate\Http\Request;

class StudentController extends Controller
{
    public function __construct(private StudentRepository $students) {}

    public function index(Request $request)
    {
        $students = $this->students->filter(
            $request->only(['search', 'status', 'graduation_year', 'class_id', 'section_id']),
            $request->input('per_page', 15)
        );

        return response()->json($students);
    }

    public function show(int $id)
    {
        return response()->json($this->students->findWithProfile($id));
    }

    public function store(StoreStudentRequest $request)
    {
        $student = $this->students->createWithUser($request->validated());

        return response()->json($student->load('user'), 201);
    }

    public function update(UpdateStudentRequest $request, int $id)
    {
        $student = $this->students->findWithUser($id);

        $this->students->updateWithUser($student, $request->validated());

        return response()->json($student->load('user'));
    }

    public function destroy(int $id)
    {
        $this->students->delete($this->students->findWithUser($id));

        return response()->json(['message' => 'Student deleted successfully.']);
    }
}
