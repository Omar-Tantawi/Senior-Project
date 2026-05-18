<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\SchoolYear\StoreSchoolYearRequest;
use App\Http\Requests\SchoolYear\UpdateSchoolYearRequest;
use App\Models\SchoolYear;

class SchoolYearController extends Controller
{
    public function index()
    {
        return response()->json(SchoolYear::latest('schoolyearid')->get());
    }

    public function store(StoreSchoolYearRequest $request)
    {
        return response()->json(SchoolYear::create($request->validated()), 201);
    }

    public function show(int $id)
    {
        return response()->json(SchoolYear::with('classes.sections')->findOrFail($id));
    }

    public function update(UpdateSchoolYearRequest $request, int $id)
    {
        $schoolYear = SchoolYear::findOrFail($id);
        $schoolYear->update($request->validated());

        return response()->json($schoolYear);
    }

    public function destroy(int $id)
    {
        SchoolYear::findOrFail($id)->delete();

        return response()->json(['message' => 'School year deleted successfully.']);
    }
}
