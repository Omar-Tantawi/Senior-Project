<?php

namespace App\Http\Requests\Assessment;

use Illuminate\Foundation\Http\FormRequest;

class StoreAssessmentRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'subject_id'         => 'required|exists:subjects,id',
            'section_id'         => 'required|exists:section,section_id',
            'title'              => 'required|string|max:255',
            'createdbyteacherid' => 'required|exists:teachers,id',
            'assessmenttype'     => 'required|in:exam,quiz,assignment,project,other',
            'date'               => 'required|date',
            'maxscore'           => 'required|numeric|min:1',
        ];
    }
}
