<?php

namespace App\Http\Requests\Student;

use Illuminate\Foundation\Http\FormRequest;

class UpdateStudentRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        $userId = $this->route('id')
            ? \App\Models\Student::findOrFail($this->route('id'))->user_id
            : null;

        return [
            'name'            => 'sometimes|string|max:255',
            'email'           => "sometimes|email|unique:users,email,{$userId}",
            'phone'           => 'nullable|string|max:20',
            'date_of_birth'   => 'nullable|date',
            'gender'          => 'nullable|in:male,female,other',
            'address'         => 'nullable|string',
            'enrollment_date' => 'nullable|date',
            'graduation_year' => 'nullable|integer|min:1900|max:2100',
            'status'          => 'nullable|in:active,graduated,transferred,withdrawn',
            'is_active'       => 'nullable|boolean',
        ];
    }
}
