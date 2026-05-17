<?php

namespace App\Http\Requests\VacationRequest;

use Illuminate\Foundation\Http\FormRequest;

class StoreVacationRequestRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'start_date' => 'required|date|after_or_equal:today',
            'end_date'   => 'required|date|after_or_equal:start_date',
        ];
    }
}
