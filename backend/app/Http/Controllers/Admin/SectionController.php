<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Section\StoreSectionRequest;
use App\Http\Requests\Section\UpdateSectionRequest;
use App\Models\Section;
use Illuminate\Http\Request;

class SectionController extends Controller
{
    public function index(Request $request)
    {
        $query = Section::with('schoolClass.schoolYear');

        if ($request->filled('class_id')) {
            $query->where('class_id', $request->class_id);
        }

        return response()->json($query->get());
    }

    public function store(StoreSectionRequest $request)
    {
        $section = Section::create($request->validated());

        return response()->json($section->load('schoolClass.schoolYear'), 201);
    }

    public function show(int $id)
    {
        return response()->json(Section::with('schoolClass.schoolYear')->findOrFail($id));
    }

    public function update(UpdateSectionRequest $request, int $id)
    {
        $section = Section::findOrFail($id);
        $section->update($request->validated());

        return response()->json($section->load('schoolClass.schoolYear'));
    }

    public function destroy(int $id)
    {
        Section::findOrFail($id)->delete();

        return response()->json(['message' => 'Section deleted successfully.']);
    }
}
