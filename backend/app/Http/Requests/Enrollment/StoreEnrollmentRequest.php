<?php

namespace App\Http\Requests\Enrollment;

use Illuminate\Foundation\Http\FormRequest;

class StoreEnrollmentRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'student_id' => 'required|exists:students,id',
            'section_id' => 'required|exists:section,section_id',
            'status'     => 'nullable|in:active,completed,dropped',
        ];
    }
}
