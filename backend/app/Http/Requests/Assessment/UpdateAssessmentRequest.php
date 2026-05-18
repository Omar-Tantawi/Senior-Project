<?php

namespace App\Http\Requests\Assessment;

use Illuminate\Foundation\Http\FormRequest;

class UpdateAssessmentRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'title'          => 'sometimes|string|max:255',
            'assessmenttype' => 'sometimes|in:exam,quiz,assignment,project,other',
            'date'           => 'sometimes|date',
            'maxscore'       => 'sometimes|numeric|min:1',
        ];
    }
}
