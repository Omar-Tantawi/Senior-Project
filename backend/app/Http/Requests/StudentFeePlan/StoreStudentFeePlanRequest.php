<?php

namespace App\Http\Requests\StudentFeePlan;

use Illuminate\Foundation\Http\FormRequest;

class StoreStudentFeePlanRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'student_id'  => 'required|exists:students,id',
            'feeplan_id'  => 'required|exists:feeplan,feeplan_id',
            'paid_amount' => 'nullable|numeric|min:0',
            'due_date'    => 'nullable|date',
            'notes'       => 'nullable|string|max:500',
        ];
    }
}
