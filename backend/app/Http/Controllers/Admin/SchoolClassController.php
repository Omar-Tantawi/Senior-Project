<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\SchoolClass\StoreSchoolClassRequest;
use App\Http\Requests\SchoolClass\UpdateSchoolClassRequest;
use App\Models\SchoolClass;
use Illuminate\Http\Request;

class SchoolClassController extends Controller
{
    public function index(Request $request)
    {
        $query = SchoolClass::with('schoolYear');

        if ($request->filled('schoolyearid')) {
            $query->where('schoolyearid', $request->schoolyearid);
        }

        return response()->json($query->get());
    }

    public function store(StoreSchoolClassRequest $request)
    {
        $class = SchoolClass::create($request->validated());

        return response()->json($class->load('schoolYear'), 201);
    }

    public function show(int $id)
    {
        return response()->json(SchoolClass::with(['schoolYear', 'sections'])->findOrFail($id));
    }

    public function update(UpdateSchoolClassRequest $request, int $id)
    {
        $class = SchoolClass::findOrFail($id);
        $class->update($request->validated());

        return response()->json($class->load('schoolYear'));
    }

    public function destroy(int $id)
    {
        SchoolClass::findOrFail($id)->delete();

        return response()->json(['message' => 'Class deleted successfully.']);
    }
}
