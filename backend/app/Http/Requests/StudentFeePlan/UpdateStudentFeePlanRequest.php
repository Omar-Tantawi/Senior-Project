<?php

namespace App\Http\Requests\StudentFeePlan;

use Illuminate\Foundation\Http\FormRequest;

class UpdateStudentFeePlanRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'paid_amount' => 'sometimes|numeric|min:0',
            'due_date'    => 'sometimes|nullable|date',
            'notes'       => 'sometimes|nullable|string|max:500',
        ];
    }
}
